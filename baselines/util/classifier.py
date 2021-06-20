import logging

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, model_from_json

from baselines.util.baselines_util import get_sbert_body_scores

NUM_SENTENCES = 4


def create_classifier(train_labels, train_embeddings, vclaim_embeddings):
    # Define the model
    model = Sequential()
    model.add(Dense(20, input_dim=NUM_SENTENCES, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Compute class weights
    total = len(train_labels.reshape((-1, 1)))
    pos = train_labels.reshape((-1)).sum()
    neg = total - pos

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    # Obtain the embeddings and scores to train the MLP (top-4 sentences per each article)
    train_embeddings = get_sbert_body_scores(train_embeddings, vclaim_embeddings, NUM_SENTENCES)

    logging.info(f"Train embeddings shape-{train_embeddings.reshape((-1, NUM_SENTENCES))}")
    logging.info(f"Train labels shape-{train_labels.reshape((-1, 1))}")

    model.fit(train_embeddings.reshape((-1, NUM_SENTENCES)),
              train_labels.reshape((-1, 1)),
              epochs=15,
              batch_size=2048,
              class_weight=class_weight)

    return model


def predict(model, iclaim_embeddings, vclaim_embeddings, iclaims, vclaims_list):
    test_scores = get_sbert_body_scores(iclaim_embeddings, vclaim_embeddings, NUM_SENTENCES)
    predictions = model.predict(test_scores.reshape((-1, NUM_SENTENCES)))

    return map_predictions(predictions, iclaims, vclaims_list)


def map_predictions(predictions, iclaims, vclaims_list):
    splitted_predictions = predictions.reshape(len(iclaims), len(vclaims_list))
    all_scores = {}
    for i, iclaim in enumerate(iclaims):
        iclaim_score = {}
        results = splitted_predictions[i]
        for j, result in enumerate(results):
            vclaim_id = vclaims_list[j]['vclaim_id']
            iclaim_score[vclaim_id] = result
        all_scores[iclaim[0]] = sorted(list(iclaim_score.items()), key=lambda x: x[1], reverse=True)

    return all_scores


def load_classifier(args, train_labels, train_encodings, vclaim_encodings):
    if args.model_path:
        json_file = open(args.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights(args.weights_path)
        logging.info(f"Loaded model from {args.weights_path}")
    else:
        classifier = create_classifier(train_labels, train_encodings, vclaim_encodings)
        if args.store_model:
            model_json = classifier.to_json()
            # TODO: Fix this so that it works without previously created model directory
            with open("model/classifier.json", "w") as json_file:
                json_file.write(model_json)
            # Serialize weights to HDF5
            classifier.save_weights("model/classifier.h5")
            logging.info("Saved model to disk")

    return classifier
