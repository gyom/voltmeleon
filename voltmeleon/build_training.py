
# TODO : import the needed modules

# TODO : Before going here, you should to implement build_model.build_submodel properly.

def build_training( cg, error_rate, cost, step_rule,
                    dataset_hdf5_file=None,
                    batch_size=256, dropout_bis=None, nbr_epochs=1, saving_path=None,
                    server_sync_extension=None,
                    server_sync_initial_read_extension=None,
                    checkpoint_interval_nbr_batches=10, diagnostic_output=None):

    # TODO : remove diagnostic_output if we're done with it

    # ici ou avant
    if dropout_bis is not None:
        # apply a second level of dropout
        inputs = VariableFilter(roles=[INPUT])(cg.variables)
        cg_dropout = cg
        for d in dropout_bis:
            for input_ in inputs:
                # TODO : check that it's indeed "_apply_args_0" that we want
                if (input_.name == d+"_apply_input_") or (input_.name == d+"_apply_args_0"):
                    print "Applying dropout %f to variable %s." % (dropout_bis[d], input_.name)
                    cg_dropout = apply_dropout(cg_dropout, [input_], dropout_bis[d])
                #else:
                #    print "Not appying any dropout to variable %s." % input_.name

        # we can do several dropout
        cg = cg_dropout
    
    train_set = H5PYDataset(dataset_hdf5_file, which_set='train')


    data_stream = DataStream.default_stream(
            #train_set, iteration_scheme=ShuffledScheme(100*batch_size, batch_size))
            train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size))
            #train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size))
    
    # monitoring the train set
    data_stream_monitor = DataStream.default_stream(
            train_set, iteration_scheme=SequentialScheme(20*batch_size, batch_size))
            #train_set, iteration_scheme=ShuffledScheme(20*batch_size, batch_size))
            #train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size))
     

    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    
    valid_set = H5PYDataset(dataset_hdf5_file, which_set='test')

    data_stream_valid =DataStream.default_stream(
            #valid_set, iteration_scheme=ShuffledScheme(10000, batch_size))
            valid_set, iteration_scheme=SequentialScheme(20*batch_size, batch_size))
            #valid_set, iteration_scheme=ShuffledScheme(20*batch_size, batch_size))

            #valid_set, iteration_scheme=ShuffledScheme(valid_set.num_examples, batch_size))
    
    """
    test_set = H5PYDataset(database['test'], which_set='test')

    data_stream_test =DataStream.default_stream(
             #test_set, iteration_scheme=ShuffledScheme(100*batch_size, batch_size))
             test_set, iteration_scheme=ShuffledScheme(test_set.num_examples, batch_size))
    

    monitor_test = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_test, prefix="test", every_n_batches=checkpoint_interval_nbr_batches)
    """

    # (elapsed_time_since_start_of_training, elapsed_time_since_1970jan01)

    timestamp_start_of_experiment = time.time()
    minibatch_timestamp = shared_floatx(np.array([0.0, timestamp_start_of_experiment], dtype=floatX))
    minibatch_timestamp.name = "minibatch_timestamp"
    def update_minibatch_timestamp(_, old_value):
        now = time.time()
        return np.array([now-timestamp_start_of_experiment, now], dtype=floatX)
    minibatch_timestamp_extension = SharedVariableModifier(minibatch_timestamp, update_minibatch_timestamp)

    monitor_valid = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_valid, prefix="test", every_n_batches=checkpoint_interval_nbr_batches)

    monitor_train = DataStreamMonitoring(
        variables=[cost, minibatch_timestamp], data_stream=data_stream_monitor, prefix="train", every_n_batches=checkpoint_interval_nbr_batches)
        #variables=[cost,diagnostic_output], ...


    extensions = [  monitor_valid,
                    monitor_train, 
                    FinishAfter(after_n_epochs=nbr_epochs),
                    Printing(every_n_batches=checkpoint_interval_nbr_batches),
                    minibatch_timestamp_extension
                  ]

    if server_sync_extension is not None:
        extensions.append(server_sync_extension)

    if server_sync_initial_read_extension is not None:
        extensions.append(server_sync_initial_read_extension)
    else:
        print "WARNING : You are not using an extension to read the parameters from the server."

    if saving_path is not None:
        extensions += [Dump(os.path.join(saving_path, "blocks_logging"))]

    main_loop = MainLoop(data_stream=data_stream,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)

    # TODO : maybe we'll return other things that make sense to have
    return main_loop


