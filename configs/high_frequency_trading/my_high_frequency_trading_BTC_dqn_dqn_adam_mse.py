task_name = "high_frequency_trading"
dataset_name = "BTC"
optimizer_name = "adam"
loss_name = "mse"
auxiliry_loss_name = "KLdiv"
net_name = "high_frequency_trading_dqn"
agent_name = "ddqn"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
]

batch_size = 512
data = dict(
    type="MyHighFrequencyTradingDataset",
    data_path="data/high_frequency_trading/small_BTC",
    train_path="data/high_frequency_trading/small_BTC/train.csv",
    valid_path="data/high_frequency_trading/small_BTC/valid.csv",
    test_path="data/high_frequency_trading/small_BTC/test.csv",
    test_dynamic_path="data/high_frequency_trading/small_BTC/Market_Dynamics_Model/BTC/test_labeled_slice_and_merge_model_3dynamics_minlength48_quantile_labeling.csv",
    tech_indicator_list=[
        # "id",
        # "symbol",
        # "exchange",
        # "datetime",
        # "name",
        "volume",
        "turnover",
        "open_interest",
        "last_price",
        "last_volume",
        "limit_up",
        "limit_down",
        "open_price",
        "high_price",
        "low_price",
        "pre_close",
        "bid_price_1",
        "bid_price_2",
        "bid_price_3",
        "bid_price_4",
        "bid_price_5",
        "ask_price_1",
        "ask_price_2",
        "ask_price_3",
        "ask_price_4",
        "ask_price_5",
        "bid_volume_1",
        "bid_volume_2",
        "bid_volume_3",
        "bid_volume_4",
        "bid_volume_5",
        "ask_volume_1",
        "ask_volume_2",
        "ask_volume_3",
        "ask_volume_4",
        "ask_volume_5",
        # "localtime",
    ],
    transcation_cost=0,
    backward_num_timestamp=1,
    max_holding_number=0.01,
    num_action=11,
    max_punish=1e12,
    episode_length=14400,
)

environment = dict(type="MyHighFrequencyTradingEnvironment")
train_environment = dict(type="MyHighFrequencyTradingTrainingEnvironment")

agent = dict(
    type="HighFrequencyTradingDDQN",
    auxiliary_coffient=512,
    reward_scale=2**0,
    repeat_times=1,
    gamma=0.99,
    batch_size=64,
    clip_grad_norm=3.0,
    soft_update_tau=0,
    state_value_tau=5e-3,
)
trainer = dict(
    type="HighFrequencyTradingTrainer",
    epochs=10,
    work_dir=work_dir,
    seeds=12345,
    batch_size=512,
    horizon_len=512,
    buffer_size=1e5,
    num_threads=8,
    if_remove=False,
    if_discrete=True,
    if_off_policy=True,
    if_keep_save=True,
    if_over_write=False,
    if_save_buffer=False,
)
loss = dict(type="HFTLoss", ada=1)
optimizer = dict(type="Adam", lr=0.001)
act = dict(type="HFTQNet", state_dim=66, action_dim=11, dims=16, explore_rate=0.01)
cri = None