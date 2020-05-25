import torch.nn  as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def discrepancyLoss(pred):
    a = pred[:,1]
    b,ind = torch.max(pred[:,1:],dim=1,keepdim=True)
    return (1.-(a-b)*(a-b)).mean()


class CosLoss(nn.Module):
    def __init__(self, nc=2):
        super(CosLoss,self).__init__()
        self.nc = nc

    def forward(self, input_np, target):
        one_hot = target.cpu()
        batchsize = target.shape[0]
        c = torch.zeros([batchsize,batchsize]).cuda()
        for i in range(self.nc):
            labels_equal = torch.eq(target,torch.tensor(i).long().cuda())
            mask = labels_equal.view(1,labels_equal.shape[0])
            mask = mask*mask.t()
            mask = mask.float()
            num = torch.sum(labels_equal).float() + 0.00000001
            mask = mask/num
            c += mask
        c = c.mm(input_np)
        torch_cos = nn.CosineSimilarity(dim = 1,eps = 1e-6)
        loss = 1 - torch_cos(input_np,c)
        return loss.mean()


class focal_loss1(nn.Module):
    def __init__(self, num_classes=2):
        super(focal_loss1, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1, 1)
        inputs = torch.cat([1 - inputs, inputs], dim=1)
        alpha = torch.tensor([[0.2], [0.8]])
        gamma = 2
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs.reshape(N, C)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        ids = ids.to(device='cuda:0', dtype=torch.int64)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)
        if inputs.is_cuda and not alpha.is_cuda:
            alpha = alpha.cuda()
        alpha = alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p
        loss = batch_loss.mean()
        return loss


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, classes, labels,reconstructions=0,images=0):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        # print(images.shape,reconstructions.shape)
        labels = labels.float()
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        # assert torch.numel(images) == torch.numel(reconstructions)
        # images = images.view(reconstructions.size()[0], -1)
        # reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        # return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
        return (margin_loss) / labels.size(0)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum().mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)



def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=8, bsize=256, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = torch.float(y_true)
        repeat_op = [torch.range(0, N).view(N,1) for i in range(N)]
        repeat_op = torch.cat(repeat_op,dim = 1).float()
        repeat_op_sq = (repeat_op - repeat_op.T())**2
        weights = repeat_op_sq / ((N - 1.) ** 2)*1.

        pred_ = y_pred ** y_pow

        pred_norm = pred_ / (eps + torch.sum(pred_, dim = 1,keepdim=True))
        hist_rater_a = torch.sum(pred_norm)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T(), y_true)
        bsize = y_true.size(0)
        nom = torch.sum(weights * conf_mat)
        denom = torch.sum(weights * torch.matmul(
            hist_rater_a.view(N, 1), hist_rater_b.view(1, N)) / (bsize*1.))

        return nom / (denom + eps)