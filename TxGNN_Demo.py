from txgnn import TxData, TxGNN, TxEval

TxData = TxData(data_folder_path = '/n/scratch3/users/k/kh278/kg/')
TxData.prepare_split(split = 'complex_disease', seed = 42, no_kg = False)

TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN'
              )

# to load a pretrained model: 
# TxGNN.load_pretrained('./model_ckpt')

TxGNN.model_initialize(n_hid = 100, 
                      n_inp = 100, 
                      n_out = 100, 
                      proto = True,
                      proto_num = 3,
                      attention = False,
                      sim_measure = 'all_nodes_profile',
                      bert_measure = 'disease_name',
                      agg_measure = 'rarity',
                      num_walks = 200,
                      walk_mode = 'bit',
                      path_length = 2)

## here we did not run this, since the output is too long to fit into the notebook
TxGNN.pretrain(n_epoch = 2, 
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)

## here as a demo, the n_epoch is set to 30. Change it to n_epoch = 500 when you use it
TxGNN.finetune(n_epoch = 30, 
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20)

TxGNN.save_model('./model_ckpt')
TxGNN.load_pretrained('./model_ckpt')

TxGNN.train_graphmask(relation = 'indication',
                      learning_rate = 3e-4,
                      allowance = 0.005,
                      epochs_per_layer = 3,
                      penalty_scaling = 1,
                      valid_per_n = 20)

output = TxGNN.retrieve_save_gates('./model_ckpt')
TxGNN.save_graphmask_model('./graphmask_model_ckpt')

from txgnn import TxEval
TxEval = TxEval(model = TxGNN)

# evaluate individual diseases
result = TxEval.eval_disease_centric(disease_idxs = [12661.0, 11318.0], 
                                     relation = 'indication', 
                                     save_result = False)

# evaluate the entire test set
result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
                                     show_plot = False, 
                                     verbose = True, 
                                     save_result = True,
                                     return_raw = False)

TxEval.retrieve_disease_idxs_test_set('indication')
