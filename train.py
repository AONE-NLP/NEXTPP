import argparse

from tpp.config_factory import Config
from tpp.runner import Runner


def main(ex_dir,ex_id):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default=ex_dir,
                        help='Dir of configuration yaml.')

    parser.add_argument('--experiment_id', type=str, required=False, default=ex_id,
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

    model_runner = Runner.build_from_config(config)

    model_runner.run()


if __name__ == '__main__':
       
    main('configs/exconfig_NEXTPP.yaml',"Earthquake")
       