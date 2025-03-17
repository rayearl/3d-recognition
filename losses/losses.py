import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureTransformRegularizer(nn.Module):
    """
    Regularization for the feature transform matrix
    """
    def __init__(self):
        super(FeatureTransformRegularizer, self).__init__()

    def forward(self, trans):
        """
        Input:
            trans: B x k x k feature transform matrix
        Return:
            Loss = |I - A*A.T|_F^2
        """
        d = trans.size()[1]
        I = torch.eye(d, device=trans.device)[None, :, :]
        batch_size = trans.size()[0]
        I = I.repeat(batch_size, 1, 1)
        product = torch.bmm(trans, trans.transpose(2, 1))
        loss = torch.mean(torch.norm(I - product, dim=(1, 2)))
        return loss

class TripletLoss(nn.Module):
    """
    Triplet loss for face recognition
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: anchor embeddings (B x D)
            positive: positive embeddings (B x D)
            negative: negative embeddings (B x D)
        Returns:
            triplet loss
        """
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        neg_dist = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)

class ArcMarginProduct(nn.Module):
    """
    ArcFace loss implementation 
    
    References:
    - https://arxiv.org/pdf/1801.07698.pdf
    - https://github.com/deepinsight/insightface/
    """
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(math.pi - margin))
        self.mm = torch.sin(torch.tensor(math.pi - margin)) * margin

    def forward(self, input, label=None):
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if label is None:
            return cosine * self.scale
            
        # For training with labels
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output

def get_center_loss(embeddings, centers, targets):
    """
    Center loss for feature learning
    
    Args:
        embeddings: feature embeddings (B x D)
        centers: class centers (num_classes x D)
        targets: class labels (B)
    Returns:
        center loss
    """
    batch_size = embeddings.size(0)
    centers_batch = centers.index_select(0, targets)
    criterion = nn.MSELoss()
    center_loss = criterion(embeddings, centers_batch)
    return center_loss

def get_combined_loss(outputs, labels, embeddings, trans_feat, centers=None, 
                     lambda_t=0.001, lambda_c=0.01, use_center_loss=False):
    """
    Combined loss function for face recognition
    
    Args:
        outputs: class scores (B x num_classes)
        labels: class labels (B)
        embeddings: feature embeddings (B x D)
        trans_feat: feature transform matrix
        centers: class centers for center loss
        lambda_t: weight for feature transform regularization
        lambda_c: weight for center loss
        use_center_loss: whether to use center loss
    Returns:
        combined loss
    """
    criterion = nn.CrossEntropyLoss()
    feature_transform_regularizer = FeatureTransformRegularizer()
    
    classification_loss = criterion(outputs, labels)
    trans_loss = feature_transform_regularizer(trans_feat)
    
    loss = classification_loss + lambda_t * trans_loss
    
    if use_center_loss and centers is not None:
        center_loss = get_center_loss(embeddings, centers, labels)
        loss = loss + lambda_c * center_loss
    
    return loss, classification_loss, trans_loss
