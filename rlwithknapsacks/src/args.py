
def get_args(parser):
    # Generic hyperparamters
    parser.add_argument("--map", type=str, default='default')
    parser.add_argument("--alg", choices=['baseline', 'optimistic', 'appropo','Toc_UCRL'], default='appropo')
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=str, default='random')
    parser.add_argument("--solver", choices=['ucbvi','policy_gradient','policy_iteration', 'reinforce','value_iteration','a2c','a2c_planner'], default="a2c")
    parser.add_argument("--horizon", type=int, default=30)
    #parser.add_argument("--sampler", choices=['egreedy', 'categorical'], default='categorical')
    parser.add_argument("--output_dir", type=str, default="/fs/clip-ml/kdbrant/workspace/rlwithknapsacks/rlwithknapsacks/results")
    parser.add_argument("--env", choices=['box_gridworld', 'gridworld', 'whisky_gridworld'],                  default='gridworld')
    parser.add_argument("--budget", nargs="+", type=float, default=[0.3])
    parser.add_argument("--randomness", type=float, default=0.1)
    parser.add_argument("--value_coef", type=float, default=0.5) #check 1 too
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--actor_critic_lr", type=float, default=0.001)
    parser.add_argument("--discount_factor", type=float, default=1.0)
    parser.add_argument("--num_episodes", type=int, default=10)

    # Optimistic hyperparamters
    parser.add_argument("--conplanner_iter", type=int, default=None)
    parser.add_argument("--bonus_coef", type=float, default=0.001)
    parser.add_argument("--planner", choices=['value_iteration', 'a2c'], default='value_iteration')
    parser.add_argument("--optimistic_lambda_lr", type=float, default=None)
    parser.add_argument("--optomistic_reset", choices=['warm-start', 'scratch', 'continue'], default='scratch')

    #A2C_Planner_hyperparameters
    parser.add_argument("--num_fic_episodes", type=int, default=1)

    # Appropo hyperparameters
    parser.add_argument("--mx_size", type=float, default=20.0)
    parser.add_argument("--proj_lr", type=float, default=None)

    # Toc_UCRL hyperparameters
    parser.add_argument("--UCRL_coeff", nargs="+",type=float, default=[0.0001,0])

    # Baseline hyperparameter
    parser.add_argument("--baseline_lambda_lr", type=float, default=None)
    args = parser.parse_args()

# Setting default parametes for griworld
def set_gridworld_defaults(args):
    if args.baseline_lambda_lr is None:
        args.baseline_lambda_lr = 0.001

    if args.conplanner_iter is None:
        args.conplanner_iter = 10

    if args.optimistic_lambda_lr is None:
        args.optimistic_lambda_lr = 0.2

    if args.proj_lr is None:
        args.proj_lr = 1.0

    if args.alg == 'appropo':
        args.num_episodes = 20
    elif args.alg == 'baseline':
        args.num_episodes = 1

# Setting the default parameters for box_gridworld
def set_boxgridworld_defaults(args):
    if args.baseline_lambda_lr is None:
        args.baseline_lambda_lr = 0.01

    if args.conplanner_iter is None:
        args.conplanner_iter = 10

    if args.optimistic_lambda_lr is None:
        args.optimistic_lambda_lr = 10

    if args.proj_lr is None:
        args.proj_lr = 10

    if args.alg == 'appropo':
        args.num_episodes = 25
    elif args.alg == 'baseline':
        args.num_episodes = 1
