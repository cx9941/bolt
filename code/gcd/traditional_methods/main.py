from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
import yaml
import sys

def apply_config_updates(args, config_dict, parser):
    """
    使用配置字典中的值更新 args 对象，同时进行类型转换。
    命令行中显式给出的参数不会被覆盖。
    """
    type_map = {action.dest: action.type for action in parser._actions}
    for key, value in config_dict.items():
        if f'--{key}' not in sys.argv and hasattr(args, key):
            expected_type = type_map.get(key)
            if expected_type and value is not None:
                try:
                    value = expected_type(value)
                except (TypeError, ValueError):
                    pass
            setattr(args, key, value)
    

if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    parser = init_model()
    # 1. 新增 --config 参数
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    # 2. 如果提供了 config 文件，则加载并应用它
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # 应用 YAML 配置，这会作为默认值
        apply_config_updates(args, yaml_config, parser)

        if 'dataset_specific_configs' in yaml_config:
            dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
            apply_config_updates(args, dataset_configs, parser)
    data = Data(args)


    print('Pre-training begin...')
    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    print('Pre-training finished!')

    print('Evaluation begin...')
    manager_p.evaluation(args, data)
    print('Evaluation finished!')

    manager_p.save_results(args)
    
