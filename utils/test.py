# import torch
#
# a = torch.max(torch.tensor([[[0.2335, 0.2455]],[[0.2329, 0.2569]]]),torch.tensor([[[0.2351, 0.4843]],[[0.7066, 0.4175]]])
# print(a)


# from pytorch_pretrained_bert.tokenization import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# sen1 = "books about bears"
# sen2 = "bottom right white couch"
# sen3 = "the handles of the slide"
#
# tokens1 = tokenizer.tokenize(sen1)
# ids1 = tokenizer.convert_tokens_to_ids(tokens1)
# print(tokens1)
# print(ids1)
#
#
# tokens2 = tokenizer.tokenize(sen2)
# ids2 = tokenizer.convert_tokens_to_ids(tokens2)
# print(tokens2)
# print(ids2)
#
#
# tokens3 = tokenizer.tokenize(sen3)
# ids3 = tokenizer.convert_tokens_to_ids(tokens3)
# print(tokens3)
# print(ids3)
import cv2
import numpy as np
# 将src中的影像粘贴至dst
src = cv2.imread('src.jpg')
dst = cv2.imread('dst.jpg')
mask = np.zeros(src.shape[:2], dtype=np.uint8)
#这个数值是src中需要被抠图的位置
mask[200:700, 600:800] = 255
#决定粘贴到的位置（在dst图像中的位置）
center, _ = cv2.minEnclosingCircle(np.array([[dst.shape[1] // 2, dst.shape[0] // 2]], dtype=np.int32))
center = tuple(map(int, center))  # 将center转换为整数类型
# Perform seamless cloning
output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
# Display result
cv2.imshow('Output', output)
cv2.waitKey(0)
