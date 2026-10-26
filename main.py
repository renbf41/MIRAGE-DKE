import os
import json
import time
import torch
import argparse
import numpy as np
from data_loader import DataLoader
from models import DistanceAwareSemanticRetriever, MIRAGERecommender
from trainer_retriever import DistanceAwareRetrieverTrainer
from trainer_gnn import GNNTrainer
from trainer_batched import BatchedTrainer
from materialize import ViewMaterializer, save_views, load_views
from rerank import rerank_all
from metrics import BeyondAccuracy
from utils import set_seed, ensure_dir, save_results


def load_retriever(data_loader, args, save_path, device):
    retriever = DistanceAwareSemanticRetriever(
        embedding_dim=data_loader.embedding_dim,
        top_k=args.top_k,
        num_negatives=256,
    ).to(device)
    name = 'retriever_best.pth' if os.path.exists(os.path.join(save_path, 'retriever_best.pth')) else 'retriever.pth'
    retriever.load_state_dict(torch.load(os.path.join(save_path, name), map_location=device))
    return retriever


def build_recommender(data_loader, device):
    return MIRAGERecommender(
        data_loader.n_users,
        data_loader.n_items,
        embedding_dim=64,
        intent_dim=data_loader.embedding_dim,
    ).to(device)


def run_retriever_and_gnn(args, data_loader, save_path, device):
    retriever = DistanceAwareSemanticRetriever(
        embedding_dim=data_loader.embedding_dim,
        top_k=args.top_k,
        num_negatives=256,
    ).to(device)
    gnn = build_recommender(data_loader, device)

    if args.stage in ['retriever', 'all']:
        print('\n--- Starting Distance-Aware Semantic Retriever Training (Stage 1 & 2) ---')
        retriever_trainer = DistanceAwareRetrieverTrainer(data_loader, retriever, lr=args.lr, top_k=args.top_k)

        best_recall = 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            loss, recall = retriever_trainer.train_epoch()
            print(f'Epoch {epoch + 1}/{args.epochs} | Loss: {loss:.4f} | Recall: {recall:.4f}')

            if recall > best_recall:
                best_recall = recall
                patience_counter = 0
                retriever_trainer.save_model(os.path.join(save_path, 'retriever_best.pth'))
                print(f'Saved best retriever model with Recall: {best_recall:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        if os.path.exists(os.path.join(save_path, 'retriever_best.pth')):
            retriever.load_state_dict(torch.load(os.path.join(save_path, 'retriever_best.pth')))
        else:
            retriever_trainer.save_model(os.path.join(save_path, 'retriever.pth'))

    if args.stage in ['gnn', 'all']:
        print('\n--- Starting PGAM Training (Stage 3) ---')

        if args.stage == 'gnn':
            if os.path.exists(os.path.join(save_path, 'retriever_best.pth')):
                retriever.load_state_dict(torch.load(os.path.join(save_path, 'retriever_best.pth')))
            else:
                retriever.load_state_dict(torch.load(os.path.join(save_path, 'retriever.pth')))

        gnn_trainer = GNNTrainer(data_loader, retriever, gnn, lr=args.lr)

        best_metric = 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            loss = gnn_trainer.train_epoch()
            print(f'Epoch {epoch + 1}/{args.epochs} | BPR Loss: {loss:.4f}')

            metrics = gnn_trainer.evaluate(data_loader.val_data, k_list=[20])
            current_metric = metrics['Recall@20']
            print(f'Validation Recall@20: {current_metric:.4f}')

            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                torch.save(gnn.state_dict(), os.path.join(save_path, 'gnn_best.pth'))
                print(f'Saved best GNN model with Recall@20: {best_metric:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        if os.path.exists(os.path.join(save_path, 'gnn_best.pth')):
            gnn.load_state_dict(torch.load(os.path.join(save_path, 'gnn_best.pth')))

        print('\n--- Starting Evaluation ---')
        metrics = gnn_trainer.evaluate(data_loader.test_data, k_list=[10, 20, 40])
        for k, v in metrics.items():
            print(f'Test {k}: {v:.4f}')

        save_results('MIRAGE', args.dataset, metrics)


def run_materialize(args, data_loader, save_path, views_path):
    print('\n--- Materializing Intent Views ---')
    retriever = load_retriever(data_loader, args, save_path, 'cpu')
    start = time.time()
    views = ViewMaterializer(data_loader, retriever, args.top_k).materialize(single_intent=args.mode == 'sim_only')
    save_views(views, views_path)
    print(f'Materialized views of {len(views)} users in {time.time() - start:.1f}s -> {views_path}')


def run_train(args, data_loader, views_path, model_path):
    print('\n--- Training on Materialized Views ---')
    model = build_recommender(data_loader, args.device)
    trainer = BatchedTrainer(data_loader, model, load_views(views_path), lr=args.train_lr, batch_size=args.batch_size,
                             negatives=args.negatives, item_layers=args.item_layers, sampling=args.sampling,
                             device=args.device)
    best_metric, best_state, patience_counter = -1.0, None, 0
    for epoch in range(args.epochs):
        start = time.time()
        loss = trainer.train_epoch()
        current_metric = trainer.evaluate(data_loader.val_data, k_list=(20,))['Recall@20']
        print(f'Epoch {epoch + 1}/{args.epochs} | BPR Loss: {loss:.4f} | Validation Recall@20: {current_metric:.4f} | {time.time() - start:.1f}s')
        if current_metric > best_metric:
            best_metric, patience_counter = current_metric, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    model.load_state_dict(best_state)
    metrics = trainer.evaluate(data_loader.test_data, k_list=(10, 20, 50))
    for k, v in metrics.items():
        print(f'Test {k}: {v:.4f}')


def run_evaluate(args, data_loader, views_path, model_path):
    print('\n--- Evaluating Materialized MIRAGE ---')
    model = build_recommender(data_loader, args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    trainer = BatchedTrainer(data_loader, model, load_views(views_path), item_layers=args.item_layers, device=args.device)
    for k, v in trainer.evaluate(data_loader.test_data, k_list=(10, 20, 50)).items():
        print(f'Test {k}: {v:.4f}')

    targets = {}
    for u, items in data_loader.test_data.items():
        valid = {int(i) for i in items if 0 <= int(i) < data_loader.n_items}
        if valid and (int(u), 0) in trainer.subgraphs.users:
            targets[int(u)] = valid
    users = np.array(sorted(targets))
    scores = trainer.scores(list(users))
    if args.lam > 0:
        texts = np.load(os.path.join(data_loader.dataset_dir, 'text_embeddings.npz'))
        lists, intents = rerank_all(scores, texts['intent'][users], texts['item'], lam=args.lam, n=args.n,
                                    pool=args.pool, gamma=args.gamma, cap=args.cap)
        names = np.array(['Similarity', 'Complementarity', 'Diversity'])
        print(f'Intent attribution shares: ' + ', '.join(f'{name} {np.mean(intents == k):.3f}' for k, name in enumerate(names)))
    else:
        lists = np.argsort(-scores, axis=1)[:, :args.n]
    for k, v in BeyondAccuracy(data_loader.ui_graph_raw).evaluate(users, lists, targets, k=args.n).items():
        print(f'Test {k}: {v:.4f}')


def run_update(args, data_loader, save_path, views_path):
    print('\n--- Incremental Maintenance of Intent Views ---')
    views = load_views(views_path)
    materializer = ViewMaterializer(data_loader, load_retriever(data_loader, args, save_path, 'cpu'), args.top_k)
    interactions = json.load(open(args.interactions))
    sizes, times = [], []
    for u, items in interactions.items():
        for item in items:
            start = time.time()
            candidate_sets, subgraphs = materializer.insert(views, int(u), int(item), single_intent=args.mode == 'sim_only')
            times.append(time.time() - start)
            sizes.append((len(candidate_sets), len(subgraphs)))
    save_views(views, views_path)
    sizes = np.asarray(sizes)
    print(f'Inserted {len(times)} interactions | updated views per insertion: {sizes[:, 0].mean():.1f} | '
          f'affected subgraphs: {sizes[:, 1].mean():.1f} | time per insertion: {np.mean(times):.3f}s')


def main():
    parser = argparse.ArgumentParser(description='MIRAGE: Multi-Intent Reasoning and Adaptive Graph Embedding')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('DATA_DIR', '.'), help='Data directory')
    parser.add_argument('--dataset', type=str, default='netflix_data', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--top_k', type=int, default=1000, help='Top K retrieval')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['retriever', 'gnn', 'all', 'materialize', 'train', 'evaluate', 'update'], help='Stage to run')
    parser.add_argument('--mode', type=str, default='multi', choices=['multi', 'sim_only'], help='Intent queries of the views')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for batched training')
    parser.add_argument('--train_lr', type=float, default=0.005, help='Learning rate on materialized views')
    parser.add_argument('--batch_size', type=int, default=64, help='Users per mini-batch')
    parser.add_argument('--negatives', type=int, default=1, help='Negative items per user')
    parser.add_argument('--item_layers', type=int, default=3, help='Propagation layers of item representations')
    parser.add_argument('--sampling', type=str, default='random', choices=['random', 'last'], help='Positive item sampling')
    parser.add_argument('--lam', type=float, default=0.0, help='Exploration parameter of intent-coverage re-ranking')
    parser.add_argument('--n', type=int, default=20, help='List length')
    parser.add_argument('--pool', type=int, default=200, help='Candidate pool size of the re-ranker')
    parser.add_argument('--gamma', type=float, default=1.0, help='Sharpening exponent of intent-item affinities')
    parser.add_argument('--cap', type=float, default=0.1, help='Cap of intent-item affinities')
    parser.add_argument('--interactions', type=str, default='', help='JSON file of new interactions for the update stage')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_loader = DataLoader(args.data_dir, args.dataset, device=device)
    save_path = os.path.join(args.data_dir, 'saved_models', args.dataset)
    ensure_dir(save_path)
    views_path = os.path.join(save_path, f'views_{args.mode}_k{args.top_k}.pkl')
    model_path = os.path.join(save_path, f'mirage_materialized_{args.mode}.pth')

    if args.stage in ['retriever', 'gnn', 'all']:
        run_retriever_and_gnn(args, data_loader, save_path, device)
    elif args.stage == 'materialize':
        run_materialize(args, data_loader, save_path, views_path)
    elif args.stage == 'train':
        run_train(args, data_loader, views_path, model_path)
    elif args.stage == 'evaluate':
        run_evaluate(args, data_loader, views_path, model_path)
    elif args.stage == 'update':
        run_update(args, data_loader, save_path, views_path)


if __name__ == '__main__':
    main()
