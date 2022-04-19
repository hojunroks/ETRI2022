import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

def dfToTensor(df, list_attributes):
    # 한 사람의 전체 label dataframe과 index tensor로 만들고 싶은 attribute들의 리스트를 입력해주면 각각의 index tensor들이 담긴 list를 반환
    attribute_tensors = []
    for attribute in list_attributes:
        type_attributes = list(df[attribute].drop_duplicates())
        tensor = torch.Tensor([x for x in df.apply(lambda x: type_attributes.index(x[attribute]), axis=1)]).to(torch.int64)
        attribute_tensors.append(tensor)
    return attribute_tensors

def indexToOneHot(attribute_tensors):
    # index tensor들이 담긴 list를 입력해주면 one hot으로 변환한 tensor들을 담은 list를 반환
    return([F.one_hot(x) for x in attribute_tensors])

def top_k(logits, y, k : int = 1):
    """
    logits : (bs, n_labels)
    y : (bs,)
    """
    labels_dim = 1
    assert 1 <= k <= logits.size(labels_dim)
    k_labels = torch.topk(input = logits, k = k, dim=labels_dim, largest=True, sorted=True)[1]

    # True (#0) if `expected label` in k_labels, False (0) if not
    a = ~torch.prod(input = torch.abs(y.unsqueeze(labels_dim) - k_labels), dim=labels_dim).to(torch.bool)
    
    # These two approaches are equivalent
    if False :
        y_pred = torch.empty_like(y)
        for i in range(y.size(0)):
            if a[i] :
                y_pred[i] = y[i]
            else :
                y_pred[i] = k_labels[i][0]
        #correct = a.to(torch.int8).numpy()
    else :
        a = a.to(torch.int8)
        y_pred = a * y + (1-a) * k_labels[:,0]
        #correct = a.numpy()

    acc = accuracy_score(y_pred, y)*100
    return acc

def evaluate(model, test_data, test_label):
    model.eval()
    test_predict = model(test_data)
    pred_act, pred_emotion = test_predict

    k_list = [1, 3, 5]
    accu_list = []
    
    for k in k_list:
        accu_list.append(top_k(pred_act.cpu(), test_label.cpu(), k))
    return accu_list

def hourToLabel(hour):
    if hour<=9:
        return "Dawn"
    elif hour<=12:
        return "Morning"
    elif hour<=18:
        return "Afternoon"
    else:
        return "Evening"
