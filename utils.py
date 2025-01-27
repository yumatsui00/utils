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

#コンテキストを左右１つの単語とする。コンテキストとして共起する単語の頻度→共起行列
def create_co_matrix(corpus, vocub_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocub_size, vocub_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

def convert_one_hot(corpus, vocab_size):
    '''コーパスをone_hot表現に変換する
    返り値：
    コーパスが一次元の場合→(N, vocab_size)
    ２次元→（N, C, vocab_size）
    '''
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    return one_hot