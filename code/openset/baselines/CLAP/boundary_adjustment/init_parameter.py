import argparse

def init_model():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default='./outputs/openset/clap', help="The unified output directory for all experiment artifacts.")
    parser.add_argument("--fold_idx", default=0, type=int, help="The index of the fold for cross-validation.")
    parser.add_argument("--fold_num", default=5, type=int, help="The total number of folds for cross-validation.")

    parser.add_argument("--data_dir", default='../data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    
    parser.add_argument("--save_results_path", type=str, default='outputs', help="the path to save results")
    
    parser.add_argument("--pretrain_dir", default='models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 
    
    parser.add_argument("--bert_model", default="../pretrained_model/bert-base-uncased", type=str, help="The path for the pre-trained bert model.")
    
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT")

    parser.add_argument("--save_model", action="store_true", help="save trained-model")

    parser.add_argument("--save_results", action="store_true", help="save test results")

    parser.add_argument("--dataset", default="banking", type=str,
                        help="The name of the dataset to train selected")
    
    parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
    parser.add_argument("--cluster_num_factor", default=1.0, type=float, help="The number of known classes")
    
    parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")

    parser.add_argument("--method", type=str, default=None, help="which method to use")
    
    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
    
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--lr", default=2e-5, type=float,
                        help="The learning rate of BERT.")    

    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.") 
    
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=256, type=int,
                        help="Batch size for evaluation.")    
    
    parser.add_argument("--wait_patient", default=10, type=int,
                        help="Patient steps for Early Stop.")    

    parser.add_argument("--lr_boundary", type=float, default=0.05, help="The learning rate of the decision boundary.")
    

    # Added for the new method
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--safe1", type=float, default=0.1)
    parser.add_argument("--safe2", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--train_from_scratch", action="store_true")
    parser.add_argument("--draw_train", action="store_true")
    parser.add_argument("--do_bert_output_norm", action="store_true")
    parser.add_argument("--aug_gen", action="store_true")
    parser.add_argument("--neg_times", type=int, default=1)
    parser.add_argument("--neg_from_gen", type=bool, default=False)

    
    return parser
