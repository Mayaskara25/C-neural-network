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

    matrix other_op_mat(matrix a, float (*fn)(float))
        Return a new matrix with fn applied to every element.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    float find_max(matrix a)
        Return the maximum element in a.
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
        parameters. Takes ownership of the network pointer.
        Return: pointer to heap-allocated TrainedModel.
        Must free_trained_model() the result.
        Do NOT free the network separately.

    void free_trained_model(TrainedModel *model)
        Free the internal NeuralNetwork (via free_neural_network) and
        the TrainedModel struct.

    void normalize_dataset(Dataset *ds, TrainedModel *model)
        Scale dataset inputs and outputs to [0, 1] by dividing by the
        per-dataset maximum. Stores input_max and output_max in model
        for denormalization during predict. Modifies dataset in-place.

    void train(TrainedModel *model, DataLoader *dl, int epochs, float lr)
        Run mini-batch gradient descent loop: for each epoch, iterate
        batches, zero accumulators, forward/backward/accumulate per
        example, then update_weights_batch. Prints loss every 100 epochs.

    matrix predict(TrainedModel *model, matrix X)
        Normalize X, run forward_network, denormalize the output.
        Return: matrix (by value, owns malloc'd data).
        Must free_mat() the result.

    void save_model(TrainedModel *model, char *path)
        Write model architecture, weights, and biases to a binary file.
        NOTE: known bugs in save/load -- see source comments.

    NeuralNetwork* load_model_network(char *path, TrainedModel **out_model)
        Read model from a binary file, reconstructing the NeuralNetwork
        and TrainedModel.
        Return: pointer to NeuralNetwork (also stored in *out_model).
        *out_model receives a new TrainedModel.
        Both must eventually be freed via free_trained_model(*out_model).
        NOTE: known bugs in save/load -- see source comments.

    ActivationID get_activationId(float (*activation)(float))
        Map an activation function pointer to its ActivationID enum value.

    void get_activation_fxn(ActivationID id,
                            float (**activation)(float),
                            float (**derivative)(float))
        Map an ActivationID enum back to the corresponding activation and
        derivative function pointers.

TO BE IMPLEMENTED
    1. Per-column normalization
       The current normalize_dataset() divides by the global maximum
       across the entire dataset. This should be changed to per-column
       (per-feature) normalization so each input/output feature is scaled
       independently to [0, 1].

    2. Terminal commands
       A command-line interface should be added so the user can run:
           ./nn train <dataset.csv>
           ./nn <modelpath> predict <testdata.csv>
       This requires CSV I/O, command-line argument parsing, and wiring
       into the existing train/predict pipeline.

    3. Training progress bar
       The current train() loop prints loss every 100 epochs. This should
       be replaced with a terminal loading bar that updates every epoch,
        showing: [========>         ] 45%  Epoch 450/1000  Loss: 0.0234

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
        ACTIVATION_SIGMOID = 0
        ACTIVATION_RELU    = 1
        ACTIVATION_TANH    = 2
        ACTIVATION_LINEAR  = 3

    TrainedModel (TRAINER)
        NeuralNetwork *network  — wrapped neural network
        float input_max         — maximum input value (for normalization)
        float output_max        — maximum output value (for normalization)
