import torch
import time
import argparse
from torch.utils.data import DataLoader
from train import *
from evaluation import *
from models import *
from utils import *
from dataset import *
import os
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

eval_dict = {
    'eval_for_tail': eval_for_tail
}

class Experiment:
    def __init__(self, config, name):
        self.model_name = config.get('model_name')
        self.train_conf = config.get('train')
        self.eval_conf = config.get('eval')
        self.dataset = Dataset(config.get('dataset'))
        self.re_name = name
        config['entity_cnt'] = len(self.dataset.data['entity'])
        config['relation_cnt'] = len(self.dataset.data['relation'])
        config['data'] = self.dataset.data['train']
        self.model, self.device = init_model(config)
        logging.info('Prior knowledge training completed')
        self.eval_func = eval_dict[self.eval_conf.get('eval_func')]
        if self.model_name in ['ConvD']:
            self.train_func = train_without_label
            self.output_func = output_eval_tail

        else:
            logging.error(f'Could not find any training function for model={self.model_name}')
        opt_conf = config.get('optimizer')
        if opt_conf.get('algorithm') == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        elif opt_conf.get('algorithm') == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        elif opt_conf.get('algorithm') == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        else:
            logging.error('Could not find corresponding optimizer for algorithm={}'.format(opt_conf.get('algorithm')))
        self.save_model_path = config.get('save_model_path')
    
    def train_and_eval(self):
        train_loader = DataLoader(self.dataset.data['train'], self.train_conf.get('batch_size'), shuffle=self.train_conf.get("shuffle"), drop_last=False)
        if self.dataset.data['valid']:
            valid_loader = DataLoader(self.dataset.data['valid'], self.eval_conf.get('batch_size'), shuffle=False, drop_last=False)
        if self.dataset.data['test']:
            test_loader = DataLoader(self.dataset.data['test'], self.eval_conf.get('batch_size'), shuffle=False, drop_last=False)
        for epoch in range(self.train_conf.get('epochs')):
            logging.info('Start training epoch: %d' % (epoch + 1))
            start_time = time.time()
            epoch_loss = self.train_func(train_loader, self.model, self.optimizer, self.device)
            end_time = time.time()
            mean_loss = np.mean(epoch_loss)
            print('[Epoch #%d] training loss: %f - training time: %.2f seconds' % (epoch + 1, mean_loss, end_time - start_time))

            if self.eval_conf.get('do_validate') and (epoch + 1) % self.eval_conf.get('valid_steps') == 0:
                print(f'--- epoch #{epoch + 1} valid ---')
                logging.info('Start evaluation of validation data')
                self.model.eval()
                with torch.no_grad():
                    eval_results = self.eval_func(valid_loader, self.model, self.device, self.dataset.data, self.eval_conf.get('scoring_desc'))
                    self.output_func(eval_results, 'validation', data = self.re_name)
        if self.eval_conf.get('do_test'):
            print(f'--- test ---')
            logging.info('Start evaluation on test data')
            self.model.eval()
            with torch.no_grad():
                eval_results = self.eval_func(test_loader, self.model, self.device, self.dataset.data, self.eval_conf.get('scoring_desc'))
                self.output_func(eval_results, 'test', data = self.re_name)
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
            logging.info('Created output directory {}'.format(self.save_model_path))
        torch.save(self.model, f'{self.save_model_path}/{self.model_name}_{self.dataset.name}.ckpt')
        logging.info('Finished! Model saved')
        return eval_results

def search_canshu(config_copy):
    search_grid = {
        "emb_dim" : [50, 100, 150, 200, 300, 400],
        "conv_out_channels" : [50, 100, 150, 200, 300, 400],
        "conv_kernel_size" : [[3, 3], [4, 4], [5, 5], [6, 6]],
        "input_dropout" : [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature_map_dropout" : [0.1, 0.2, 0.3, 0.4, 0.5],
        "hidden_dropout" : [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    ans = 50
    res = 0
    uuu = {}
    for key, value in search_grid.items():
        res = 0
        for item in search_grid[key]:
            config = copy.deepcopy(config_copy)
            config["model_hyper_params"][key] = item
            print(config)
            experiment = Experiment(config, str(ans))
            ans_res = experiment.train_and_eval()
            if res < np.array(ans_res[4]).mean():
                res = np.array(ans_res[4]).mean()
                uuu[key] = item
            ans = ans + 1
    print(uuu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge graph inference arguments.')
    parser.add_argument('-c', '--config', dest='config_file', help='The path of configuration json file.')
    args = parser.parse_args()
    print(args)
    name = os.path.splitext(os.path.basename(args.config_file))[0]

    config = load_json_config(args.config_file)
    print(config)

    search_canshu(config)

    # experiment = Experiment(config, name)

    # experiment.train_and_eval()

