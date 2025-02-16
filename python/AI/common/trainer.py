import numpy as np
import matplotlib.pyplot as plt
import time

def clip_grads(grads, max_norm, epsilon=1e-6):
	'''勾配クリッピング：勾配全体がmax_normを超えた場合にクリッピング'''
	total_norm = 0
	for grad in grads:
		total_norm += np.sum(grad ** 2)
	total_norm = np.sqrt(total_norm)
	rate = max_norm / (total_norm + epsilon)
	if rate < 1:
		for grad in grads:
			grad *= rate

def remove_duplicate(params, grads):
    '''パラメータ配列中の重複する重みを一つに集約し、その重みに対応する勾配を加算する'''
    params, grads = params[:], grads[:]  # copy list
    while True:
        find_flg = False
        L = len(params)
        for i in range(0, L-1):
            for j in range(i+1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j].astype(grads[i].dtype)  # 型を統一
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T.astype(grads[i].dtype)  # 型を統一
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break
    return params, grads


class Trainer:
	def __init__(self, model, optimizer):
		self.model = model
		self.optimizer = optimizer
		self.loss_list = []
		self.eval_interval = None
		self.current_epoch = 0

	def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
		data_size = len(x)
		max_iters = data_size // batch_size
		self.eval_interval = eval_interval
		model, optimizer = self.model, self.optimizer
		total_loss = 0
		loss_count = 0

		start_time = time.time()
		for epoch in range(max_epoch):
			idx = np.random.permutation(np.arange(data_size))
			x = x[idx]
			t = t[idx]
			for iters in range(max_iters):
				start = iters * batch_size # シャッフルしてバッチ取得
				end = (iters + 1) * batch_size
				batch_x = x[start:end]
				batch_t = t[start:end]

				loss = model.forward(batch_x, batch_t)
				model.backward()
				params, grads = remove_duplicate(batch_x, batch_t)
				if max_grad is not None:
					clip_grads(grads, max_grad)
				optimizer.update(params, grads)
				total_loss += loss
				loss_count += 1

				#評価
				if (eval_interval is not None) and (iters % eval_interval) == 0:
					avg_loss = total_loss / loss_count
					elapsed_time = time.time() - start_time
					print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
					% (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
					self.loss_list.append(float(avg_loss))
					total_loss, loss_count = 0, 0
			self.current_epoch += 1

	def plot(self, ylim=None):
		x = np.arange(len(self.loss_list))
		if ylim is not None:
			plt.ylim(*ylim)
		plt.plot(x, self.loss_list, label='train')
		plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
		plt.ylabel('loss')
		plt.show()


# if __init__ == '__main__':
# 	window_size = 1
# 	hidden_size = 5
# 	batch_size = 3
# 	max_epoch = 1000

# 	text = 'you say goodbye and I say hello'
# 	corpus, word_to_id, id_to_word = preprocess(text)
# 	vocab_size = len(word_to_id)
# 	contexts, target = create_contexts_target(corpus, window_size)
# 	target = convert_one_hot(target, vocab_size)
# 	contexts = convert_one_hot(contexts, vocab_size)

# 	model = SimpleCBOW(vocab_size, hidden_size)
# 	optimizer = Adam()
# 	trainer = Trainer(model, optimizer)

# 	trainer.fit(contexts, target, max_epoch, batch_size)
# 	trainer.plot()