# Concurrent-KNN

Implementação concorrente do algoritmo KNN em linguagem C usando biblioteca Pthreads.

O algoritmo espera dois arquivos .csv, um correspondente a base de dados de treinamento e outro correspondente a base de teste.
O caminho dos arquivos é definido pelas variávels DATA_TRAIN e DATA_TEST (linhas 21 e 25 do arquivo Trabalho_170050432.c);
O número de amostras de treino e de teste é definido pelas variáveis TRAIN_SAMPLES e TEST_SAMPLES (linhas 22 e 26 o arquivo Trabalho_170050432.c), respectivamente.
O número de rótulos do dataset é definido por N_LABELS, na linha 29.

Por padrão, o algoritmo rodará em 8 threads, esse valor pode ser mudado modificando o valor da variável N_THREADS na linha 34.
