/*
Implementacão Concorrente do Algoritmo KNN
    Igor Bispo - 170050432
    Programação Concorrente

    !! É necessário ter os arquivos do dataset para executar o código !!

    https://drive.google.com/file/d/1Tjm3IzKqq9uMoqHcSsWKff5q7khairkL/view?usp=sharing

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>

// Espera que o arquivo mnist_train.csv esteja no diretório data/
const char* DATA_TRAIN = "data/mnist_train.csv";
const size_t TRAIN_SAMPLES = 60000;

// Espera que o arquivo mnist_test.csv esteja no diretório data/
const char* DATA_TEST = "data/mnist_test.csv";
const size_t TEST_SAMPLES = 10000;

// Número de labels do dataset
const size_t N_LABELS = 10;

/****/

// Número de threads
const size_t N_THREADS = 8;

/* Este vetor armazenará quais as fronteiras do vetor de amostras de treino cada
 thread do KNN executará
 */
int** TRAIN_THREAD_BOUNDS;

/* Este vetor armazenará quais as fronteiras do vetor de amostras de teste cada
 thread do KNN executará
 */
int** TEST_THREAD_BOUNDS;

// Estrutura que será passada como argumento para as funções thread_knn_predict e paralel KNN
typedef struct {
    int* sample;
    int** x_train;
    int** x_test;
    int* y_train;
    size_t* y_pred;
    int low_lim;
    int upper_lim;
    int k_neighbors;

    size_t* labels_threads;
    int* label_idx;
    pthread_mutex_t* lock_labels;
} knn_arg;


/******** Rotinas de IO ********/

const size_t SAMPLE_SIZE = 784;

// Entrada é um vetor do formato <label, 1x1, 1x2...28x28>
int parse_line (char* line, int** x, int* y) {
    int* x_out = malloc(SAMPLE_SIZE*sizeof(int));

    if (!x_out) return -1;

    char* tok  = strtok(line, ",");
    *y = atoi(tok);

    for (int i = 0; tok && i < SAMPLE_SIZE; tok = strtok(NULL, ","), i++) {
        x_out[i] = atoi(tok);
    }

    *x = x_out;

    return 0;
}

int load_csv(int*** x, int** y, const char* file_path, int n_samples) {

    FILE* file = fopen(file_path, "r");

    if (!file) {
        printf("Não foi possível carregar o arquivo de entrada %s\n", file_path);
        return -1;
    }

    size_t buff_size = 8192;
    char* buffer = calloc(buff_size, 1);

    // Ignora a primeira linha contendo o cabeçalho do arquivo CSV
    if (getline(&buffer, &buff_size, file) == -1) return -1;

    int ** x_out = malloc(n_samples*sizeof(int*));
    int* y_out = malloc(n_samples*sizeof(int));

    printf("Carregando arquivo %s ...\n", file_path);

    int i = 0;
    // Entrada é um vetor do formato <label, 1x1, 1x2...28x28>
    while (getline(&buffer, &buff_size, file) != -1 && i < n_samples) {
        if (parse_line(buffer, &x_out[i], &y_out[i]) != 0) return -1;

        i += 1;
    }

    free(buffer);

    *x = x_out;
    *y = y_out;

    return 0;
}

void free_memory(int** ptr, size_t n) {
    for (size_t i = 0; i < n; i++)
        free(ptr[i]);
    free(ptr);
}

/******** FIM Rotinas de IO ********/

/******** Rotinas Aux. KNN ********/

// Calcula a distância euclidiana (L2) entre x1 e x2
double l2_distance(int* x1, int* x2, int size) {
    long l2 = 0;

    for (int i = 0; i < size; i++) {
        l2 += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }

    return sqrt((double) l2);
}


// Ordena as primeiras N posições do vetor de chave "key" usando como referência o vetor "x"
void partial_key_sort(int* x, int* key, int size, int N) {

    for (int i = 0; i < N; i++){
        int smallest_idx = i;
        
        for (int j = i + 1; j < size; j++) {
            if (x[j] < x[smallest_idx])
                smallest_idx = j;
        }

        // Trocando key
        int temp = key[i];
        key[i] = key[smallest_idx];
        key[smallest_idx] = temp;

        // Trocando x
        temp = x[i];
        x[i] = x[smallest_idx];
        x[smallest_idx] = temp;
    }
}

// Avalia a acurácia do algoritmo de decisão
double evaluate_acc(int* y_test, size_t* y_pred, int size) {
    int correct = 0;
    
    for (int i = 0; i < size; i++) {
        if (y_test[i] == y_pred[i])
            correct += 1;
    }

    return (double) correct / (double) size;
}
/******** FIM Rotinas Aux. KNN ********/

/******** Rotinas KNN ********/

/*
    Esta função calcula quais as fronteiras de operação para cada umas das threads

    Todas as threads receberão uma fronteira do mesmo tamanho, exceto a última
    que pode receber uma fronteira maior caso o tamanho "size" do vetor não seja
    divisível por N_THREADS.

    O resultado retornado será uma matriz cujo as linhas uma dupla com o formato (low, high)
    Em que "low" é o limite inferior da fronteira e "high" é o limite superior.
*/
int** calculate_thread_bounds(int size) {
    int** bounds = malloc(sizeof(int*) * size);

    int quocient = (int) size / N_THREADS;
    int remainder = size - quocient * N_THREADS;

    for (int i = 0;i < N_THREADS; i++) {
        bounds[i] = malloc(sizeof(int) * 2);
    }

    for (int i = 0;i < N_THREADS -1; i++) {
        bounds[i][0] = i*quocient;
        bounds[i][1] = (i+1)*quocient;
    }

    bounds[N_THREADS-1][0] = (N_THREADS-1)*quocient;
    bounds[N_THREADS-1][1] = (N_THREADS)*quocient + remainder;  

    return bounds;
}

/* 
  Esta função roda uma instância do KNN entre as fronteiras arg->low_lim e arg->upper_lim 
  da matriz arg->x_train e do vetor arg->y_train. Classifica a amostra arg->sample.

  O resultado da execução será salvo em arg->labels_thread na posição arg->label_idx e
  label_idx será incrementado em 1
*/

void* thread_knn_predict(void* arg) {
    knn_arg* params = (knn_arg*) arg;

    size_t v_size = params->upper_lim - params->low_lim;

    int* distances = malloc(sizeof(int)*v_size);
    int* neighbors = malloc(sizeof(int)*v_size);

    // Calculando a distância L2 entre x e cada uma das amostras de x_train entre "low_lim" e "upper_lim"
    // E populando o vetor "neighbors"
    int idx = 0;
    for (int i = params->low_lim; i < TRAIN_SAMPLES && i < params->upper_lim; i++) {
        distances[idx] = l2_distance(params->sample, params->x_train[i], SAMPLE_SIZE);
        neighbors[idx] = i;
        idx += 1;
    }

    // Ordenando vetor de vizinhos "neighbors" usando com referência "distances"
    // São ordenadas apenas as primeiras "arg->n_neighbors" posições para evitar cálculos desnecessários
    partial_key_sort(distances, neighbors, v_size, params->k_neighbors);

    // Calculando o histograma de labels dos k_neighbors vizinhos 
    int* histogram = calloc(N_LABELS, sizeof(int));

    for (int i = 0;i < params->k_neighbors; i++) {
        histogram[ params->y_train[neighbors[i]]] += 1;
    }

    // Determinando qual a label com maior frequência em "histogram" para prever a label de "x"
    size_t label = 0;

    for (int i = 1;i < N_LABELS; i++) {
        if (histogram[i] > histogram[label])
            label = i;
    }

    free(histogram);
    free(distances);
    free(neighbors);

    pthread_mutex_lock(params->lock_labels);
    params->labels_threads[*params->label_idx] = label;
    *params->label_idx += 1;
    pthread_mutex_unlock(params->lock_labels);

    pthread_exit(0);
}

/*
  Esta função executa o algoritmo KNN em uma amostra usando N_THREADS paralelas

  Cada uma das threads executará o algoritmo entre as fronteiras
  estabelecidas no vetor TRAIN_THREAD_BOUNDS

  Os valores de cada thread serão salvos em args->labels_threads e depois será feita
  uma "votação" para determinar o valor final do classificador

*/


size_t paralel_knn_predict(int* x, int** x_train, int* y_train, int k_neighbors) {
    pthread_t prediction_threads[N_THREADS];

    size_t* labels_threads = malloc(N_THREADS * sizeof(size_t));

    int* label_idx = malloc(sizeof(int));
    *label_idx = 0;

    pthread_mutex_t* mutex_ptr = malloc(sizeof(pthread_mutex_t));
    if (pthread_mutex_init(mutex_ptr, NULL) != 0){
        printf("Erro ao inicializar mutex\n");
        exit(1);
    }

    // Executa o algoritmo KNN sobre um subconjunto das amostras de treino
    for (int i = 0; i < N_THREADS; i++) {
        knn_arg* args = malloc(sizeof(knn_arg));
	
	// Define a amostra que será classificada pelas threads de thread_knn_predict
        args->sample = x;

        args->x_train = x_train;
        args->y_train = y_train;

	// Define os limites do vetores x_train e y_train em que cada thread irá operar
        args->low_lim = TRAIN_THREAD_BOUNDS[i][0];
        args->upper_lim = TRAIN_THREAD_BOUNDS[i][1];

        args->k_neighbors = k_neighbors;

	// O vetor de predições é o índice da última adição serão compartilhados entre as threads
        args->labels_threads = labels_threads;
        args->label_idx = label_idx;

	// O mutex é compartilhado entre as threads
        args->lock_labels = mutex_ptr;

        pthread_create(&prediction_threads[i], NULL, &thread_knn_predict, args);
    }

    // Espera todas as threads finalizarem antes de continuar
    for (int i = 0; i < N_THREADS; i++) {
        pthread_join(prediction_threads[i], NULL);
    }

    /*
       Faz uma votação entre os predições obtidas por cada uma das threads
       O valor mais comum é dado como a predição final.
    */

    int* histogram = calloc(N_LABELS, sizeof(int));
    for (int i = 0; i < N_THREADS; i++) {
        histogram[labels_threads[i]] += 1;
    }

    size_t label = 0;
    for (int i = 1; i < N_LABELS; i++) {
        if (histogram[i] > histogram[label])
            label = i;
    }

    free(histogram);
    free(labels_threads);

    return label;
}

/* 
   Esta função executará paralel_knn_predict em cada uma das "test_samples" amostras da matriz "x_test" 
   entre as fronteiras estabelecidas no vetor TEST_THREAD_BOUNDS.


   Os valores de predição serão salvos em params->y_pred na posição da amostra de teste respectiva
*/

void* paralel_knn(void* arg) {
    knn_arg* params = (knn_arg*) arg;

    size_t n_samples = params->upper_lim - params->low_lim;

    printf("\tThread executando KNN com %ld amostras de teste ...\n", n_samples);

    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = params->low_lim; i < TEST_SAMPLES && i < params->upper_lim ; i++) {

        /* Não é necessário um lock aqui porque cada uma das threads de "paralel_knn"
           acessa uma fronteira diferente da outra
        */
        params->y_pred[i] = paralel_knn_predict(params->x_test[i], params->x_train, params->y_train, params->k_neighbors);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    
    printf("\tThread finalizada em %lf s\n", (double) elapsed);
}

/* 
   Esta função chamará "paralel_knn" de forma concorrente em cada uma das fronteiras
   estabelecidas no vetor TEST_THREAD_BOUNDS.

   O resultado da execução de "paralel_knn" será atualizado no vetor "y_pred" por "paralel_knn"
*/

size_t* knn_classifier(int** x_train, int* y_train, int** x_test, int n_samples, int k_neighbors) {

    pthread_t knn_threads[N_THREADS];

    size_t* y_pred = malloc(n_samples*sizeof(size_t));

    printf("Executando KNN com %ld amostras de treino e %d amostras de teste usando %ld threads...\n", TRAIN_SAMPLES, n_samples, N_THREADS);

    // Executa N_THREADS paralelas de classificadores KNN sobre as amostras de teste
    for (int i = 0;i < N_THREADS;i++) {
        knn_arg* args = malloc(sizeof(knn_arg));

        args->x_train = x_train;
        args->y_train = y_train;
        args->x_test = x_test;

	// Vetor em que será salvo as predições feitas por paralel_knn
        args->y_pred = y_pred;

	// Define os limites de operação de cada thread de paralel_knn
        args->low_lim = TEST_THREAD_BOUNDS[i][0];
        args->upper_lim = TEST_THREAD_BOUNDS[i][1];

        args->k_neighbors = k_neighbors;

        pthread_create(&knn_threads[i], NULL, &paralel_knn, args);
    }


    // Espera todas as threads finalizarem antes de continuar
    for (int i = 0; i < N_THREADS; i++) {
        pthread_join(knn_threads[i], NULL);
    }

    printf("\n");

    return y_pred;
}


/******** FIM Rotinas KNN ********/

int main() {
    int** x_train;
    int** x_test;

    int* y_train;
    int* y_test;

    // Carrega o arquivo csv com as amostras de treino
    load_csv(&x_train, &y_train, DATA_TRAIN, TRAIN_SAMPLES);

    // Carrega o arquivo csv com as amostras de test
    load_csv(&x_test, &y_test, DATA_TEST, TEST_SAMPLES);

    // Inicializa os valores de fronteira do vetor de treino que cada thread acessará no KNN
    TRAIN_THREAD_BOUNDS = calculate_thread_bounds(TRAIN_SAMPLES);

    // Inicializa os valores de fronteira do vetor de test que cada thread acessará no KNN
    TEST_THREAD_BOUNDS = calculate_thread_bounds(TEST_SAMPLES);

    size_t* y_pred = knn_classifier(x_train, y_train, x_test, TEST_SAMPLES, 3);

    double acc = evaluate_acc(y_test, y_pred, TEST_SAMPLES);

    printf("Acurácia obtida: %lf%%\n.", acc * 100);


    // Liberando memória
    free_memory(x_train, TRAIN_SAMPLES);

    free_memory(x_test, TEST_SAMPLES);

    free(y_train);
    free(y_test);

    free_memory(TRAIN_THREAD_BOUNDS, N_THREADS);
    free_memory(TEST_THREAD_BOUNDS, N_THREADS);


}
