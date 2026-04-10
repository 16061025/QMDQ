from distutils.core import setup_keywords

import torch

from agents.VanillaDQN import *


class VanillarhoFixOverlapDQN(VanillaDQN):
    '''
    Implementation of Vanilla DQN with only replay buffer (no target network)
    '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.K = cfg['agent']['networks_num']  # number of networks
        self.Ratio_overlap = cfg['agent']['Ratio_overlap']
        if cfg['agent']['M'] is None:
            self.M = math.ceil((1 + self.Ratio_overlap) * self.K / 2)
        else:
            self.M = cfg['agent']['M']
        # Create k different: Q value network and Optimizer
        self.Q_net = [None] * self.K
        self.optimizer = [None] * self.K
        for i in range(self.K):
            self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
            self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
                                                                               **cfg['optimizer']['kwargs'])

    # def __init__(self, cfg):
    #     super().__init__(cfg)
    #     self.K = cfg['agent']['networks_num']  # number of networks
    #     self.Ratio_overlap = cfg['agent']['Ratio_overlap']
    #     if cfg['agent']['M'] is None:
    #         self.M = math.ceil((1 + self.Ratio_overlap) * self.K / 2)
    #     else:
    #         self.M = cfg['agent']['M']
    #     # Create k different: Q value network and Optimizer
    #     self.Q_net = [None] * self.K
    #     self.optimizer = [None] * self.K
    #     self.Q_net[0] = self.createNN(cfg['env']['input_type']).to(self.device)
    #     self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(),
    #                                                                        **cfg['optimizer']['kwargs'])
    #     for i in range(1, self.K):
    #         self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
    #         self.Q_net[i].load_state_dict(self.Q_net[0].state_dict())
    #         self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
    #                                                                            **cfg['optimizer']['kwargs'])


    def get_sel_est_indices(self):



        if np.random.random() >= 0.5:
            self.sel_indices = np.arange(self.M)
            self.est_indices = np.arange(start=self.K - self.M, stop=self.K)
            self.mere_sel_start = 0
            self.mere_sel_end = self.K-self.M - 1
            self.mere_est_start = self.M
            self.mere_est_end = self.K - 1
            self.mix_start = self.K - self.M
            self.mix_end = self.M-1


        else:
            self.est_indices = np.arange(self.M)
            self.sel_indices = np.arange(start=self.K - self.M, stop=self.K)
            self.mere_est_start = 0
            self.mere_est_end = self.K-self.M - 1
            self.mere_sel_start = self.M
            self.mere_sel_end = self.K - 1
            self.mix_start = self.K - self.M
            self.mix_end = self.M-1



        return self.sel_indices, self.est_indices

    def update_target_net(self):
        pass


    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
        sel_indices, est_indices = self.get_sel_est_indices()

        with torch.no_grad():
            q_target = self.compute_q_target(batch) #batch*sel

        action = batch.action.long().unsqueeze(1)

        for i in self.sel_indices:
            q = self.Q_net[i](batch.state).gather(1, action).squeeze()

            loss = self.loss(q, q_target)
            # Take an optimization step
            self.optimizer[i].zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[i].parameters(), self.gradient_clip)
            self.optimizer[i].step()
            if self.show_tb:
                self.logger.add_scalar(f'Loss', loss.item(), self.step_count)

    def compute_q_target(self, batch):
        with torch.no_grad():
            i = self.mere_sel_start
            mere_sel_q_sa = 0
            while i<=self.mere_sel_end:
                q_net = self.Q_net[i]
                q_next_sa = q_net(batch.next_state)
                mere_sel_q_sa += q_next_sa
                i += 1

            i= self.mix_start
            mix_q_sa = 0
            while i<=self.mix_end:
                q_net = self.Q_net[i]
                q_next_sa = q_net(batch.next_state)
                mix_q_sa += q_next_sa
                i += 1

            i = self.mere_est_start
            mere_est_q_sa = 0
            while i<=self.mere_est_end:
                q_net = self.Q_net[i]
                q_next_sa = q_net(batch.next_state)
                mere_est_q_sa += q_next_sa
                i += 1

            sel_q_sa = mere_sel_q_sa + mix_q_sa
            optimal_action = torch.argmax(sel_q_sa, dim=1) #batch

            average_est_q_sa = (mere_est_q_sa + mix_q_sa)/self.M
            average_q_next = average_est_q_sa.gather(1, optimal_action.unsqueeze(-1)).squeeze()

            q_target = batch.reward + self.discount * average_q_next * batch.mask
        return q_target


    def compute_q(self, batch):
        # Convert actions to long so they can be used as indexes
        action = batch.action.long()
        action = action.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.M)
        sel_Q_nets = self.Q_net[self.sel_indices[0]:self.sel_indices[-1] + 1]

        q_sAlla_list = []
        for q_net in sel_Q_nets:
            q_sa = q_net(batch.state)
            q_sAlla_list.append(q_sa)

        q_sAlla = torch.stack(q_sAlla_list, dim=-1)  # batch*N_action*N_sel
        q = q_sAlla.gather(1, action).squeeze(dim=1)  # batch*N_sel
        return q


    def get_action_selection_q_values(self, state):
        q_values_list = []
        for i in range(self.K):
            q_values = self.Q_net[i](state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()

        return q_values





