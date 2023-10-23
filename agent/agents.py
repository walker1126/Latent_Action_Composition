from agent.base_agent import BaseAgent
from functional.motion import get_foot_vel
import torch


class Agent2x(BaseAgent):
    def __init__(self, config, net):
        super(Agent2x, self).__init__(config, net)
        self.inputs_name = ['input1', 'input2', 'input12', 'input21']
        self.targets_name = ['target1', 'target2', 'target12', 'target21']

    def forward(self, data):
        inputs = [data[name].to(self.device) for name in self.inputs_name]
        targets = [data[name].to(self.device) for name in self.targets_name]

        # update loss metric
        losses = {}

        if self.use_triplet:
            outputs, motionvecs, staticvecs = self.net.cross_with_triplet(*inputs)
            losses['m_tpl1'] = self.triplet_weight * self.tripletloss(motionvecs[2], motionvecs[0], motionvecs[1])
            losses['m_tpl2'] = self.triplet_weight * self.tripletloss(motionvecs[3], motionvecs[1], motionvecs[0])
            losses['b_tpl1'] = self.triplet_weight * self.tripletloss(staticvecs[2], staticvecs[0], staticvecs[1])
            losses['b_tpl2'] = self.triplet_weight * self.tripletloss(staticvecs[3], staticvecs[1], staticvecs[0])
        else:
            outputs = self.net.cross(inputs[0], inputs[1])

        for i, target in enumerate(targets):
            losses['rec' + self.targets_name[i][6:]] = self.mse(outputs[i], target)

        if self.use_footvel_loss:
            losses['foot_vel'] = 0
            for i, target in enumerate(targets):
                losses['foot_vel'] += self.footvel_loss_weight * self.mse(get_foot_vel(outputs[i], self.foot_idx),
                                                                          get_foot_vel(target, self.foot_idx))

        outputs_dict = {
            "output1": outputs[0],
            "output2": outputs[1],
            "output12": outputs[2],
            "output21": outputs[3],
        }
        return outputs_dict, losses


