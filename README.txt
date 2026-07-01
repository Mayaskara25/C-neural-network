MEMORY OWNERSHIP :
anything to do with the layer is cleared by the layer ;
any other stuff is cleared by the caller;
keep an eye on what is being returned - most of the returned stuff here have pointers in them -i.e you have to look at two seperate things:
they are
1)when the caller calls a fxn that retrns a struct or anything that has something malloced ,  what happens to the memory that was previously linked to it?
  i.e create_dense_layer() makes and allocs memory for every member matrix in the layer , but the forward pass makes and puts new stuff in input , output , and z - what happens to the memeory that these
structs were linked to earlier ?- REMEBER TO FREE tHEM - i.e free_mat(layer->input); layer->input = copy_mat(input);

2)when a function returns a struct that is a member of the layer , the CALLER DOES NOT OWN THE MEMORY , matrix x = forward_pass(...) ; this returns layer->output to x , now both x and layer->outpu point to
the same memory location , so DO NOT FREE MATRIX X , just freeing the dense_layer sould free the heap pointed by x aswell.

BUILD
    make all            Build all test targets.
    make <target>       Build a single test (e.g. make trainer_test).
    make clean          Remove all built binaries.

    Available targets: trainer_test, backprop_test, layer_test,
    model_test, matrix_test, csv_test.

NAME
    c-neural-network — neural network library in C

MATRIX OPERATIONS
    matrix create_mat(int rows, int cols)
        Allocate a rows×cols matrix on the heap.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    matrix copy_mat(matrix a)
        Return a deep copy of a.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    void print_mat(matrix a)
        Print matrix a to stdout.
        No free needed.

    void input_mat(matrix a)
        Read matrix a from stdin.
        No free needed.

    void free_mat(matrix *a)
        Free a->data and zero the struct fields.

    matrix add_mat(matrix a, matrix b)
        Return element-wise sum of a and b.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    matrix sub_mat(matrix a, matrix b)
        Return element-wise difference of a and b.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    matrix scalarmul_mat(matrix a, float b)
        Return a scaled by scalar b.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    matrix mul_mat(matrix a, matrix b)
        Return the matrix (dot) product a × b.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    matrix transpose_mat(matrix a)
        Return the transpose of a.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    matrix elemul_mat(matrix a, matrix b)
        Return the element-wise (Hadamard) product.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    matrix elediv_mat(matrix a, matrix b)
        Return element-wise division a / b.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    matrix other_op_mat(matrix a, float (*fn)(float))
        Return a new matrix with fn applied to every element.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    float find_max(matrix a)
        Return the maximum element in a.
        Return: scalar float. No free needed.

    float find_max_abs(matrix a)
        Return the maximum absolute value in a.
        Return: scalar float. No free needed.

    void rand_init_mat(matrix a)
        Fill a uniformly with random values in [-1, 1].

    void zero_mat(matrix a)
        Set all elements of a to 0.

    void xavier_init_mat(matrix a, int input_size)
        Xavier/Glorot uniform initialization in [-1/sqrt(n), 1/sqrt(n)].

    void he_init_mat(matrix a, int input_size)
        He/Kaiming uniform initialization in [-sqrt(6/n), sqrt(6/n)].

ACTIVATION FUNCTIONS
    float sigmoid_activation(float a)
    float sigmoid_derivative(float a)
    float relu_activation(float a)
    float relu_derivative(float a)
    float leaky_relu_activation(float a)
    float leaky_relu_derivative(float a)
    float tanh_activation(float a)
    float tanh_derivative(float a)
    float no_activation(float a)
    float no_activ_derivative(float a)
        All return scalar floats. No free needed.

LAYER
    DenseLayer* create_dense_layer(int input_size, int output_size,
                                   float (*activation)(float),
                                   float (*derivative)(float))
        Allocate and initialize a fully-connected layer with weights, bias,
        and gradient buffers.
        Return: pointer to heap-allocated DenseLayer.
        Must free_dense_layer() the result.

    void forward_pass(matrix input, DenseLayer *layer)
        Compute z = W·x + b, then apply activation to produce output.
        Caches input, z, and output for backprop. Frees and re-allocates
        internal matrices each call.

    void backward_pass(matrix upstream_gradient, DenseLayer *layer)
        Compute delta, dW, db, and dx from the upstream loss gradient.
        Frees and re-allocates internal matrices each call.

    void zero_accumulators(DenseLayer *layer)
        Zero the accumulated gradient buffers for batch training.
        Allocates accumulated_dW and accumulated_db on first call.

    void accumulate_gradients(DenseLayer *layer)
        Add dW and db to the accumulated gradient buffers.

    void free_dense_layer(DenseLayer *layer)
        Free all 11 internal matrices and the layer struct itself.

MODEL
    NeuralNetwork* create_neural_network(void)
        Allocate an empty sequential neural network (capacity starts at 2).
        Return: pointer to heap-allocated NeuralNetwork.
        Must free_neural_network() the result.

    void add_layer(NeuralNetwork *network, DenseLayer *layer)
        Append a layer to the network. The network takes ownership; do not
        free the layer separately.

    matrix const* forward_network(matrix input, NeuralNetwork *network)
        Run forward pass through every layer in sequence.
        Return: const pointer to the last layer's output matrix.
        The caller does NOT own this memory -- it belongs to the last layer.
        Do NOT free it; it is freed when the layer is freed.

    void backward_network(matrix loss_grad, NeuralNetwork *network)
        Run backward pass through every layer in reverse order.

    void update_weights(NeuralNetwork *network, float lr)
        SGD weight update: W -= lr * dW, b -= lr * db using per-example
        gradients. Frees and re-allocates weight/bias matrices.

    void zero_network_accumulators(NeuralNetwork *nn)
        Call zero_accumulators on every layer in the network.

    void accumulate_network_gradients(NeuralNetwork *nn)
        Call accumulate_gradients on every layer in the network.

    void update_weights_batch(NeuralNetwork *nn, float lr, int batch_size)
        Batch weight update: W -= lr * accumulated_dW / batch_size,
        b -= lr * accumulated_db / batch_size. Operates in-place; no
        allocations.

    void free_neural_network(NeuralNetwork *network)
        Free every layer via free_dense_layer, then the layer array,
        then the network struct.

LOSS
    float mse(matrix y_pred, matrix y_actual)
        Compute mean squared error between prediction and target.
        Return: scalar float. No free needed.

    matrix mse_derivative(matrix y_pred, matrix y_actual)
        Compute gradient of MSE w.r.t. predictions: -(2/N)(y_actual - y_pred).
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    float mae(matrix y_pred, matrix y_actual)
        Compute mean absolute error: (1/N)summation|y_pred - y_actual|.
        Return: scalar float. No free needed.

    matrix mae_derivative(matrix y_pred, matrix y_actual)
        Compute gradient of MAE w.r.t. predictions: sign(y_pred - y_actual)/N.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    float huber(matrix y_pred, matrix y_actual)
        Compute Huber loss (delta hardcoded to 1.0): quadratic for
        |y_pred - y_actual| <= delta, linear beyond it.
        Return: scalar float. No free needed.

    matrix huber_derivative(matrix y_pred, matrix y_actual)
        Compute gradient of Huber loss w.r.t. predictions.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    float binary_cross_entropy(matrix y_pred, matrix y_actual)
        Compute binary cross-entropy loss. Predictions are clamped to
        [1e-7, 1-1e-7] to avoid log/divide-by-zero.
        NOTE: not wired into LossID/get_loss_fxn/train() yet -- implemented
        for future use. Meaningful only when the output layer uses sigmoid
        activation.
        Return: scalar float. No free needed.

    matrix binary_cross_entropy_derivative(matrix y_pred, matrix y_actual)
        Compute the raw (unfused) gradient of BCE w.r.t. predictions:
        (y_pred - y_actual) / (y_pred*(1 - y_pred)) / N. Not pre-multiplied
        by sigmoid'(z) -- backward_pass() already applies the output layer's
        activation derivative.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    void get_loss_fxn(LossID id, float (**loss)(matrix, matrix),
                      matrix (**loss_derivative)(matrix, matrix))
        Map a LossID to its loss and loss-derivative function pointers.
        Unknown ids fall back to LOSS_MSE.

DATASET
    Dataset* create_dataset(int input_size, int output_size)
        Allocate an empty dataset with dynamic capacity (starts at 8).
        Return: pointer to heap-allocated Dataset.
        Must free_dataset() the result.

    void add_example(Dataset *ds, matrix input, matrix output)
        Deep-copy and append an input-output pair. Reallocates arrays when
        full. All memory is owned by the Dataset.

    void free_dataset(Dataset *ds)
        Free every example's matrices (via free_mat), then the example
        arrays, then the Dataset struct.

DATALOADER
    DataLoader* create_dataloader(Dataset *ds, int batch_size)
        Allocate a DataLoader for shuffled mini-batch iteration.
        Return: pointer to heap-allocated DataLoader.
        Must free_dataloader() the result.
        NOTE: does NOT take ownership of the Dataset; caller must still
        free_dataset() separately.

    void reset_dataloader(DataLoader *dl)
        Reset iteration to start and reshuffle indices.

    int has_next_batch(DataLoader *dl)
        Return 1 if more batches remain, 0 otherwise.

    Batch next_batch(DataLoader *dl)
        Return the next batch. The returned Batch's .inputs and .outputs
        arrays are shallow copies into the Dataset's data.
        Return: Batch (by value). The array pointers are newly allocated
        but the matrix data inside points into the Dataset.
        Must free_batch() the result; the matrix data is freed by
        free_dataset().

    void free_batch(Batch b)
        Free b.inputs and b.outputs arrays only.
        Does NOT free the matrix data inside (owned by Dataset).

    void free_dataloader(DataLoader *dl)
        Free the indices array and the DataLoader struct.
        Does NOT free the associated Dataset.

TRAINER
    TrainedModel* create_trained_model(NeuralNetwork *network)
        Wrap a NeuralNetwork in a TrainedModel, which holds normalization
        parameters (per-feature max vectors). Takes ownership of the
        network pointer.
        Return: pointer to heap-allocated TrainedModel.
        Must free_trained_model() the result.
        Do NOT free the network separately.

    void free_trained_model(TrainedModel *model)
        Free input_max and output_max matrices, the internal NeuralNetwork
        (via free_neural_network), and the TrainedModel struct.

    void normalize_dataset(Dataset *ds, TrainedModel *model)
        Per-feature max-abs scaling to [-1, 1]. Each input/output feature
        is independently scaled by its own maximum absolute value. Stores
        per-feature max vectors as matrices (input_max, output_max) in
        model for denormalization during predict. Features with max 0 are
        left unscaled. Modifies dataset in-place.

    void train(TrainedModel *model, DataLoader *dl, int epochs, float lr, LossID loss_id)
        Run mini-batch gradient descent loop using the loss selected by
        loss_id (see LossID under STRUCT DATA MEMBERS). For each epoch,
        iterate batches, zero accumulators, forward/backward/accumulate per
        example, then update_weights_batch. Displays a progress bar with
        percentage, epoch count, and loss.

    matrix predict(TrainedModel *model, matrix X)
        Normalize X by per-feature division, run forward_network,
        denormalize the output by per-feature multiplication.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    void save_model(TrainedModel *model, char *path)
        Write model architecture, weights, biases, and per-feature
        normalization vectors to a binary file.

    NeuralNetwork* load_model_network(char *path, TrainedModel **out_model)
        Read model from a binary file, reconstructing the NeuralNetwork
        and TrainedModel with normalization vectors.
        Return: pointer to NeuralNetwork (also stored in *out_model).
        *out_model receives a new TrainedModel.
        Both must eventually be freed via free_trained_model(*out_model).

    ActivationID get_activationId(float (*activation)(float))
        Map an activation function pointer to its ActivationID enum value.

    void get_activation_fxn(ActivationID id,
                            float (**activation)(float),
                            float (**derivative)(float))
        Map an ActivationID enum back to the corresponding activation and
        derivative function pointers.

CSV
    CSV* create_csv(int rows, int cols)
        Allocate an empty CSV with rows×cols data and no header.
        Return: pointer to heap-allocated CSV.
        Must free_csv() the result.

    CSV* read_csv(const char *path, int has_header)
        Parse a CSV file into a CSV struct. If has_header is 1, the first
        line is read as column names into header[]. Data is stored as a
        flat row-major float array. Handles Windows line endings.
        Return: pointer to heap-allocated CSV, or NULL on error.
        Must free_csv() the result.

    void write_csv(const CSV *csv, const char *path)
        Write a CSV struct to a file. Writes header line if has_header is
        set. Data values formatted with %.7g precision.

    void print_csv(const CSV *csv)
        Print CSV to stdout in tab-separated format for debugging.
        No free needed.

    void free_csv(CSV *csv)
        Free data array, each header string, header array, and the CSV
        struct itself. Safe to call with NULL.

    Dataset* csv_to_dataset(const CSV *csv, int input_cols, int output_cols)
        Convert a CSV to a Dataset. The first input_cols columns become
        inputs, the last output_cols columns become outputs. Middle
        columns (if any) are silently skipped. Returns NULL if
        input_cols + output_cols > csv->cols.
        Return: pointer to heap-allocated Dataset.
        Must free_dataset() the result.

TO BE IMPLEMENTED
    1. Terminal commands
       A command-line interface should be added so the user can run:
           ./nn train <dataset.csv>
           ./nn <modelpath> predict <testdata.csv>
       CSV I/O is now available via the csv module. This requires
       command-line argument parsing and wiring into the existing
       train/predict pipeline.

STRUCT DATA MEMBERS
    matrix (MATRIX OPERATIONS)
        int rows              — number of rows
        int cols              — number of columns
        float *data           — row-major flattened array

    DenseLayer (LAYER)
        matrix weights              — weight matrix (input_size × output_size)
        matrix bias                 — bias vector (1 × output_size)
        matrix output               — cached post-activation output
        matrix input                — cached input from forward pass
        matrix z                    — cached pre-activation values
        matrix delta                — backpropagated error
        matrix dW                   — weight gradient
        matrix db                   — bias gradient
        matrix dx                   — input gradient
        matrix accumulated_dW       — accumulated weight gradient (batch)
        matrix accumulated_db       — accumulated bias gradient (batch)
        float (*activation)(float)            — activation function pointer
        float (*activation_derivative)(float) — derivative function pointer

    NeuralNetwork (MODEL)
        DenseLayer **layer    — dynamic array of layer pointers
        int num_of_layers     — number of layers added
        int capacity           — allocated capacity of the layer array

    Dataset (DATASET)
        matrix *inputs        — array of input matrices
        matrix *outputs       — array of output matrices
        int num_examples      — number of examples stored
        int capacity          — allocated capacity of the arrays
        int input_size        — dimensionality of each input
        int output_size       — dimensionality of each output

    Batch (DATALOADER)
        matrix *inputs        — array of input matrices in the batch
        matrix *outputs       — array of output matrices in the batch
        int size              — number of examples in the batch

    DataLoader (DATALOADER)
        Dataset *ds           — pointer to the source dataset
        int batch_size        — number of examples per batch
        int current_pos       — current position in the shuffled index list
        int *indices          — shuffled array of dataset indices

    ActivationID (TRAINER, enum)
        ACTIVATION_SIGMOID    = 0
        ACTIVATION_RELU       = 1
        ACTIVATION_TANH       = 2
        ACTIVATION_LINEAR     = 3
        ACTIVATION_LEAKY_RELU = 4

    LossID (LOSS, enum)
        LOSS_MSE   = 0
        LOSS_MAE   = 1
        LOSS_HUBER = 2

    TrainedModel (TRAINER)
        NeuralNetwork *network  — wrapped neural network
        matrix input_max        — per-feature max-abs vector for inputs
        matrix output_max       — per-feature max-abs vector for outputs

    CSV (CSV)
        float *data           — row-major flattened float array
        int rows              — number of data rows (excludes header)
        int cols              — number of columns
        char **header         — array of column name strings, or NULL
        int has_header        — 1 if header present, 0 if not

USAGE PIPELINE
    The functions in this library should be called in the following order
    to set up a complete training and prediction pipeline:

    1. Load data
           CSV *csv = read_csv("data.csv", 1);
           Dataset *ds = csv_to_dataset(csv, input_cols, output_cols);
           free_csv(csv);

    2. Build the network
           NeuralNetwork *nn = create_neural_network();
           add_layer(nn, create_dense_layer(input_cols, 64, relu_activation, relu_derivative));
           add_layer(nn, create_dense_layer(64, output_cols, no_activation, no_activ_derivative));

    3. Wrap in TrainedModel
           TrainedModel *model = create_trained_model(nn);

    4. Normalize the dataset
           normalize_dataset(ds, model);

    5. Create the DataLoader
           DataLoader *dl = create_dataloader(ds, batch_size);

    6. Train
           train(model, dl, epochs, learning_rate, LOSS_MSE);

    7. Predict (normalization/denormalization is handled internally)
           matrix input = create_mat(input_cols, 1);
           // ... fill input values ...
           matrix result = predict(model, input);
           // ... use result ...
           free_mat(&result);
           free_mat(&input);

    8. Save / Load
           save_model(model, "model.bin");
           // later:
           TrainedModel *loaded;
           load_model_network("model.bin", &loaded);

    9. Cleanup (in this order)
           free_dataloader(dl);
           free_dataset(ds);
           free_trained_model(model);
