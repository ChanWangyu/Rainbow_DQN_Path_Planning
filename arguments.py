def add_args(parser):
    parser.add_argument('--write', action='store_false', default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--test', action='store_true', default=False, help='True means test phase, otherwise train phase')
    parser.add_argument('--debug', action='store_true', default=False, )
    # parser.add_argument('--load_dir', type=str, help='If load model, specify the location')
    # parser.add_argument('--load_timestep', type=int, help='If load model, specify the timestep')
    parser.add_argument('--ModelIdex', type=int, default=400, help='which model to load')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', action='store_false', default=True)
    parser.add_argument('--gpu_id', type=str, default='0', help='')
    parser.add_argument('--Max_train_steps', type=int, default=50000, help='Max training steps, rllib iter=1000 等价于1.2M个ts')
    parser.add_argument('--save_interval', type=int, default=10000, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=10, help='Model evaluating interval, in steps.')
    parser.add_argument('--num_test_episode', type=int, default=1, )

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    # parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--net_width', type=int, default=128, help='Hidden net width')
    parser.add_argument('--q_lr', type=float, default=2e-4, help='Learning rate of q_net')
    # parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')

    parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory of actor and critic')
    parser.add_argument('--coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--share_parameter', action='store_true', default=False, help='Share parameter between agent or not')

    # envs
    parser.add_argument('--type2_act_dim', type=int, default=8)

    # dir
    parser.add_argument("--dataset", type=str, default='manhattan', choices=['NY', 'manhattan'])
    parser.add_argument("--config_dir", type=str, help='import which config file')
    parser.add_argument("--setting_dir", type=str, help='Will be used in noma envs')
    parser.add_argument("--output_dir", type=str, default='./results/debug', help="which fold to save under 'results/'")

    # roadmap
    parser.add_argument('--roadmap_dir', type=str, default='.')
    parser.add_argument('--gr', type=int, default=200, choices=[0, 50, 100, 200],)
    parser.add_argument('--env_type', type=str, default='default') # dijkstra
    # multi-thread
    parser.add_argument('--n_rollout_threads', type=int, default=1)

    # debug
    parser.add_argument('--reward_scale', type=float, default=1)

    parser.add_argument('--num_car', type=int, default=None)
    parser.add_argument('--num_order', type=int, default=None)

    return parser