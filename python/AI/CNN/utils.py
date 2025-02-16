import numpy as np

def conv_output_size(input_size, filter_size, stride=1, pad=0):
	return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
	"""
	畳み込み演算を効率的に行うため、画像データを２次元配列に展開
	通常 (入力画像)*(フィルター) = 出力画像
	im2col (展開後の二次元行列)*(フィルター行列) = 出力


	Parameters
	---------------
	input_data : (データ数、チャンネル、高さ、幅)の四次元配列入力データ
	filter_h : フィルター高さ
	filter_w : フィルター幅
	stride : ストライド
	pad : パディング

	Returns
	----------------
	col : 二次元配列
	"""
	N, C, H, W = input_data.shape
	out_h = (H + 2*pad - filter_h) // stride + 1
	out_w = (W + 2*pad - filter_w) // stride + 1

	img = np.pad(input_data, [(0,0), (0, 0), (pad, pad), (pad, pad)], 'constant')
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x * stride*out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

	col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
	return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]