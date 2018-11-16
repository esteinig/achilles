#!/usr/bin/env python3

from achilles.terminal.client import entry_point

entry_point()

#
# def main():
#
#     # Terminal input
#     args = Terminal().args
#
#     if args["agg"]:
#         plt.switch_backend('agg')
#
#     #####################
#     ##  TASK: create   ##
#     #####################
#
#     if args["subparser"] == "create":
#
#         # Generate data for batch-wise input into Achilles,
#         # write training and validation data to HDF5 file
#         if args["config"]:
#             # Modify arguments from config file:
#             args, uri = read_create_config(args)
#         else:
#             uri = args["uri"]
#
#         pongo = PoreMongo(uri=uri)
#         pongo.connect()
#
#         if args["ssh"]:
#             pongo.open_ssh(config_file=args["config"])
#             pongo.open_scp()
#
#         ds = AchillesDataset(poremongo=pongo)
#
#         if args["display"]:
#             ds.poremongo.display_tags()
#
#         ds.write(tags=args["tags"], data_file=args["data_file"], max_windows=args["max_windows"],
#                  max_windows_per_read=args["max_windows_per_read"], window_size=args["window_size"],
#                  window_step=args["window_step"], window_random=args["random"], window_recover=args["recover"],
#                  sample_files_per_tag=args["sample_files_per_tag"], sample_proportions=args["sample_proportions"],
#                  sample_unique=args["sample_unique"], exclude_datasets=args["exclude"], global_tags=args["global_tags"],
#                  validation=args["validation"], scale=args["scale"], chunk_size=args["chunk_size"], ssh=args["ssh"])
#
#         pongo.disconnect()
#         if args["ssh"]:
#             pongo.close_ssh()
#             pongo.close_scp()
#
#     #####################
#     ##  TASK: sample   ##
#     #####################
#
#     if args["subparser"] == "sample":
#
#         # Config for URI and argument modification:
#         if args["config"]:
#             args, uri = read_create_config(args)
#         else:
#             uri = args["uri"]
#
#         # MongoDB and SSH connections
#         pongo = PoreMongo(uri=uri)
#         pongo.connect()
#
#         if args["ssh"]:
#             pongo.open_ssh(config_file=args["config"])
#             pongo.open_scp()
#
#         sampled = []
#         labels = []
#         for label, tags in enumerate(args["tags"]):
#             sample = pongo.sample(Fast5.objects, tags=tags, limit=args["sample_files_per_tag"],
#                                   proportion=args["sample_proportions"], unique=args["sample_unique"],
#                                   exclude_name=args["exclude"], include_tags=args["include_tags"])
#             sampled += sample
#             labels += [label for _ in sample]
#
#         if args["out_dir"]:
#             pongo.copy(sampled, outdir=args["out_dir"], ncpu=args["threads"], iterate=args["iterate"])
#         else:
#             pongo.to_csv(sampled, labels=labels, out_file=args["out_file"])
#
#         pongo.disconnect()
#         if args["ssh"]:
#             pongo.close_ssh()
#             pongo.close_scp()
#
#     if args["subparser"] == "train":
#
#         # Build model
#         achilles = Achilles(data_file=args["data_file"])
#
#         if args["load"]:
#             achilles.load_model(args["load"])
#         else:
#             achilles.build(window_size=args["window_size"], activation=args["activation"],
#                            nb_residual_block=args["nb_residual_blocks"], nb_channels=args["nb_channels"],
#                            nb_rnn=args["nb_rnn"], rnn_units=args["rnn_units"], gru=args["gru"], gpu=args["gpu"],
#                            dropout=args["dropout"], rc_dropout=args["rc_dropout"], bidirectional=args["bi"])
#
#             # Compile model with loss function and optimizer
#             achilles.compile(optimizer=args["optimizer"], loss=args["loss"])
#
#         achilles.train(epochs=args["epochs"], batch_size=args["batch_size"], workers=args["threads"],
#                        run_id=args["run_id"], outdir=args["output_dir"], verbose=args["verbose"])
#
#     if args["subparser"] == "evaluate":
#
#         achilles_evaluation = AchillesEvaluation(data=args["data_files"])
#
#         achilles_evaluation.evaluate(models=args["model_files"], batch_size=args["batch_size"],
#                                      threads=args["threads"], data_path=args["data_path"],
#                                      out_file=args["output_file"])
#
#     if args["subparser"] == "pipeline":
#
#         achilles.pipeline()
#
#
#     # if args["subparser"] == "predict":
#     #
#     #     predict(fast5=args["input_files"], model=args["model_file"], window_max=args["windows"],
#     #             window_size=args["window_size"], window_step=args["window_step"],
#     #             batches=args["batches"], window_random=args["window_random"])
#     #
#     # if args["subparser"] == "pevaluate":
#     #
#     #     evaluate_predictions(dirs=args["dirs"], model=args["model_file"], window_max=args["windows"],
#     #                          window_size=args["window_size"], window_step=args["window_step"],
#     #                          batches=args["batches"], window_random=args["window_random"],
#     #                          prefix=args["prefix"], include=args["include"], exclude=args["exclude"],
#     #                          class_labels=args["labels"])
#
#
# def config():
#
#     with open(r"/home/esteinig/.keras/keras.json", "r") as keras_config:
#         config = json.load(keras_config)
#
#         config["image_dim_ordering"] = 'channel_last'
#         config["backend"] = "tensorflow"
#
#     with open(r"/home/esteinig/.keras/keras.json", "w") as keras_config:
#         json.dump(config, keras_config)
#
#
# main()
