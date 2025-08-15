import yaml
import argparse
import sys

def main():
    # 使用 parse_known_args() 来分离出我们已知的 --config 参数
    # 和所有未知的覆盖参数 (e.g., --rate 0.5 --seed 1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args, overrides = parser.parse_known_args()

    # 1. 加载基础的 YAML 配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(1)

    # 2. 应用命令行覆盖参数
    # overrides 列表会是这样的: ['--rate', '0.5', '--seed', '1']
    for i in range(0, len(overrides), 2):
        key = overrides[i].lstrip('-') # 去掉 '--'
        value = overrides[i+1]
        
        # 尝试转换值为数字类型，如果失败则保持为字符串
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        
        config[key] = value

    # 3. 将最终的配置转换为 shell 参数并输出
    shell_args = []
    if config:
        for key, value in config.items():
            shell_args.append(f'--{key}')
            shell_args.append(str(value))
            
    print(' '.join(shell_args))

if __name__ == '__main__':
    main()