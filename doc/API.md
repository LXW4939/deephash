QuantizationBase: 相当于源代码中的database，主要实现code，output





问题：

1. 不同class之间共享session以及graph？传session





Dvsq组成部分：

1. image dataset
   1. ​
2. Architecture:
   1. load_model(train model and val model) to get features
   2. train_layers, train_last_layer
   3. // Dim, lr_exponential_strategy, lr_multiply, learning_rate, decay_step
3. Cos loss
4. code database for quantization
5. CQ
   1. init(load model), deep representation (PQ init)
   2. initial_centers
   3. update_centers
   4. update_codes_batch
6. Evaluation
7. dvsq
   1. train
      1. for each batch:
         1. Load_model to get output
         2. get multiple loss and loss combination
         3. learning rate shedule and apply gradients with different lr for each layer(会不会有问题？如果把这部写到Architecture中，则无法定制，如写到dvsq中，则无法拓展，不同网络架构这部分代码会不一样)
      2. for each epoch
         1. update_codes_batch
         2. update_centers
   2. Val 
      1. For each batch in query and database
         1. pass forward to get output
         2. get cos loss
         3. feed_batch_output
      2. update_codes_batch for query and database
      3. evaluation

