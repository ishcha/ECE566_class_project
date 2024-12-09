"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
import os.path

from torch.utils.tensorboard import SummaryWriter

from utils.args import *
from utils.utils import *
import pickle
import json

from domainbed import datasets_pfl
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed_net import *

def init_clients(args_, root_path, logs_dir, save_path, dataset, hparams):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")

    # train_iterators, val_iterators, test_iterators = \
    #     get_loaders(
    #         type_=LOADER_TYPE[args_.experiment],
    #         root_path=root_path,
    #         batch_size=args_.bz,
    #         is_validation=args_.validation,
    #         dist_shift=args.dist_shift,
    #         dp = args.dp,
    #         emb_size= args_.embedding_dimension
    #     )
    
    
    # data_dir = get_data_dir(args_.experiment)
    
    # if "logs_dir" in args_:
    #     logs_dir = args_.logs_dir
    # else:
    #     logs_dir = os.path.join("logs", arguments_manager_.args_to_string())

    # if "chkpts_dir" in args_:
    #     chkpts_dir = args_.chkpts_dir
    # else:
    #     chkpts_dir = os.path.join("/data/enyij2/pfl/chkpts", arguments_manager_.args_to_string())

    train_iterators, val_iterators, test_iterators = [], [], []
    
    for env_i, env in enumerate(dataset):
        uda = []

        # split training/testing data
        out, in_ = misc.split_dataset(env,
            int(len(env)*args_.holdout_fraction),
            misc.seed_hash(args_.trial_seed, env_i))
        # split finetuning set from testing data
        # clients.append(dataset.ENVIRONMENTS[env_i])
        if env_i in args_.test_envs:
            # server = dataset.ENVIRONMENTS[env_i]
            uda, in_ = misc.split_dataset(env,
                int(len(in_)*args_.uda_holdout_fraction),
                misc.seed_hash(args_.trial_seed, env_i))
            out, in_ = misc.split_dataset(in_,
            int(len(in_)*args_.holdout_fraction),
            misc.seed_hash(args_.trial_seed, env_i))
            args_.n_target_samples = len(uda)
            print(f"number of target samples: {len(uda)}")
            # clients_size.append(len(uda))
            # server_dls['train'].append(torch.utils.data.DataLoader(
            # uda,
            # num_workers=dataset.N_WORKERS,
            # batch_size=16))
            # server_dls['test'].append(torch.utils.data.DataLoader(
            # out,
            # num_workers=dataset.N_WORKERS,
            # batch_size=64))
            # train_iterators.append(torch.utils.data.DataLoader(
            # uda,
            # num_workers=dataset.N_WORKERS,
            # batch_size=hparams['batch_size']))
            for _ in range(len(dataset)-1):
                test_iterators.append(torch.utils.data.DataLoader(
                out,
                num_workers=dataset.N_WORKERS,
                batch_size=64))
                val_iterators.append(torch.utils.data.DataLoader(
                in_,
                num_workers=dataset.N_WORKERS,
                batch_size=64))
        else:
            # clients_size.append(len(in_))
            train_iterators.append(torch.utils.data.DataLoader(
            in_,
            num_workers=dataset.N_WORKERS,
            batch_size=hparams['batch_size']))

            # test_iterators.append(torch.utils.data.DataLoader(
            # out,
            # num_workers=dataset.N_WORKERS,
            # batch_size=64))
            
            # val_iterators.append(torch.utils.data.DataLoader(
            # in_,
            # num_workers=dataset.N_WORKERS,
            # batch_size=64))


    # all_data_tensor = []
    # for cur_data in train_iterators:
    #     # print(cur_data.dataset.underlying_dataset.tensors[0].shape)
    #     cur_dataset = cur_data.dataset.underlying_dataset
    #     all_images = []
    #     for img, label in cur_dataset:
    #         all_images.append(img)
    #     all_images_tensor = torch.stack(all_images)
    #     all_data_tensor.append(all_images_tensor)
    # all_data_tensor = torch.cat(all_data_tensor, dim=0)
    # print(all_data_tensor.shape)
    # model = models.resnet18(pretrained=True)
    # model = domainbedNet(dataset.input_shape, dataset.num_classes,
    #     len(dataset), hparams)

    
    # # del model.fc
    # # all_data_tensor = all_data_tensor.view(-1,1,28,28)
    # all_data_tensor = all_data_tensor.view(-1, 3, 224, 224)
    # x = all_data_tensor
    # if all_data_tensor.shape[1] == 1:
    #     x = all_data_tensor.repeat(1, 3, 1, 1)
    # # x = model.conv1(x.float())
    # # x = model.bn1(x)
    # # x = model.relu(x)
    # # x = model.maxpool(x)
    
    # # x = model.layer1(x)
    # # x = model.layer2(x)
    # # x = model.layer3(x)
    # # x = model.layer4(x)
    # x = model.featurizer.network(x)
    # # Extract the feature maps produced by the encoder
    # encoder_output = x.squeeze()
    # print(x.shape)
    # U, S, V = torch.svd(encoder_output)
    # global PCA_V
    # PCA_V = V
    # print(PCA_V.size())
    # with open(f"data/domainbed/PCA.pkl" , 'wb') as f:
    #     pickle.dump(PCA_V, f)

    # encoder_output = encoder_output.view(encoder_output.size(0), -1)
    # pca_transformer = PCA(n_components=emb_size)
    # # Fit the PCA transformer to your data
    
    # X_pca = pca_transformer.fit_transform(encoder_output.detach().numpy())
    # # Convert the resulting principal components to a PyTorch tensor
    # projected = torch.from_numpy(X_pca).float().cuda()

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):
        # if train_iterator is None or test_iterator is None:
        #     continue
        # if

        model = domainbedNet(dataset.input_shape, dataset.num_classes,
        len(dataset), hparams)

        # print(args_.experiment)

        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                custom_model=model
            )
        # learners_ensemble.load_state("saves/femnist_unseen_base/FedGMM_sanity/global_ensemble.pt")
        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        save_path_ = os.path.join(save_path, "task_{}".format(task_id))
        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            save_path=save_path_,
            tune_locally=args_.locally_tune_clients
        )

        clients_.append(client)
    return clients_


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    if args_.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args_.algorithm, args_.dataset)
    else:
        hparams = hparams_registry.random_hparams(args_.algorithm, args_.dataset,
            misc.seed_hash(args_.hparams_seed, args_.trial_seed))
    if args_.hparams:
        hparams.update(json.loads(args_.hparams))
    

    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # assign device argument
    if torch.cuda.is_available():
        args_.device = "cuda"
    else:
        args_.device = "cpu"
    

    # assign learning rate
    args_.lr = hparams["lr"]

    # data_dir = get_data_dir(args_.experiment, unseen=False)
    # data_dir = '/data/enyij2/domainbed/data'
    save_dir = get_save_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    if args_.dataset in vars(datasets_pfl):
        dataset = vars(datasets_pfl)[args_.dataset](args_.data_dir,
            args_.test_envs, hparams)
    else:
        raise NotImplementedError

    print("==> Clients initialization..")
    clients = init_clients(args_,
                           root_path=args_.data_dir,
                           logs_dir=os.path.join(logs_dir, "train"),
                           save_path=os.path.join(save_dir, "train"),
                           dataset=dataset,
                           hparams=hparams
                           )

    print("==> Test Clients initialization..")
    test_clients = init_clients(args_, root_path=os.path.join(args_.data_dir, "test"),
                                logs_dir=os.path.join(logs_dir, "test"),
                                save_path=os.path.join(save_dir, "test"),
                                dataset=dataset,
                                hparams=hparams)
    # test_clients = None

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)
    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_model = domainbedNet(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args_.test_envs), hparams)
    
    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            client_type=CLIENT_TYPE[args_.method],
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            embedding_dim=args_.embedding_dimension,
            n_gmm=args_.n_gmm,
            custom_model=global_model
        )

    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    # global_learners_ensemble.load_state("saves/femnist_unseen_base/FedGMM_sanity/global_ensemble.pt")
    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            em_step=args.em_step,
            seed=args_.seed
        )
    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)

    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    gmm = False
    while current_round <= args_.n_rounds:
        # aggregator.update_clients()
        # aggregator.write_log()
        # raise
        if aggregator_type == "ACGcentralized":
            aggregator.mix()
        else:
            aggregator.mix()
        if current_round % 10 == 0:
            aggregator.save_state(save_dir)
            print("saved at epoch", current_round)

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round
        print(current_round)
        # if current_round > 50:
        #     gmm=False

    global_train_logger.flush()
    global_test_logger.flush()
    global_train_logger.close()
    global_test_logger.close()



if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    run_experiment(args)
