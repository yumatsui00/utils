def cos_similarity(x, y, eps=1e-8): #eps でゼロベクトルが入力された際の０による除算対策
    nx = x / np.sqrt(np.sum(x ** 2) + eps) #正規化
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    return np.dot(nx, ny) #内積が1に近いほど意味が似ている

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f'{query} is not found')
        return 
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity_matrix = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity_matrix[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity_matrix).argsort():
        if id_to_word[i] == query:
            continue
        print(f' {id_to_word[i]}: {similarity_matrix[i]}')
        count += 1
        if count >= top:
            return

#C:共起行列
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i] + eps))
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M
