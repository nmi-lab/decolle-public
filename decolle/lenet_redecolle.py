from .base_model import *
from .utils import tonp
from .lenet_decolle_model import LenetDECOLLE



class LenetREDECOLLE(LenetDECOLLE):
    def init_parameters(self, *args, **kwargs):
        super().init_parameters(*args, **kwargs)



class RecLIFLayer(BaseLIFLayer):
    NeuronState = namedtuple('NeuronState', ['P', 'Pr', 'Q', 'Qr', 'R', 'S'])
    def __init__(self, *args, **kwargs):
        super(RecLIFLayer, self).__init__(*args, **kwargs)
        
        #Make the following smarter to handle convolutional neural networks.
        self.rec_layer = nn.Linear(self.base_layer.out_features, self.base_layer.out_features, bias=False)
    
    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.rec_layer = self.rec_layer.to(device)
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.rec_layer = self.rec_layer.cpu()
        return self
    
    def init_state(self, Sin_t):
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      Pr=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      Qr=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      R =torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([out_ch] + out_shape).type(dtype).to(device))
        
    def forward(self, Sin_t, Inj=0):
        if self.state is None:
            self.init_state(Sin_t)

        #Forward traces
        state = self.state
        P = self.alpha * state.P + (1-self.alpha)*state.Q
        Q = self.beta  * state.Q + (1-self.beta) *Sin_t*self.gain

        #Recurrent traces
        Pr = self.alpha * state.Pr + (1-self.alpha)*state.Qr
        Qr = self.beta  * state.Qr + (1-self.beta) *state.S

        #Refractory
        R = self.alpharp * state.R - state.S * self.wrp

        #Membrane potential
        U = self.base_layer(P) + self.rec_layer(Pr) + R 
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P,
                                      Q=Q,
                                      Pr=Pr,
                                      Qr=Qr,
                                      R= R,
                                      S= S)
        if self.do_detach: 
            state_detach(self.state)
            
        return S, U

    def init_parameters(self, *args, **kwargs):
        self.reset_parameters(self.base_layer, *args, **kwargs)
        self.reset_rec_parameters(self.rec_layer, *args, **kwargs)

    def reset_parameters(self, layer):
        layer.reset_parameters()
        if hasattr(layer, 'out_channels'):
            if layer.bias is not None:
                layer.bias.data = layer.bias.data*((1-self.alpha)*(1-self.beta))
            layer.weight.data[:] *= 1
        elif hasattr(layer, 'out_features'): 
            if layer.bias is not None:
                layer.bias.data[:] = layer.bias.data[:]*((1-self.alpha)*(1-self.beta))
            layer.weight.data[:] *= 5e-2
        else:
            warnings.warn('Unhandled data type, not resetting parameters')

    def reset_rec_parameters(self, layer):
        layer.reset_parameters()
        if hasattr(layer, 'out_channels'):
            if layer.bias is not None:
                layer.bias.data *= 0 
            layer.weight.data *= 0 
        elif hasattr(layer, 'out_features'): 
            if layer.bias is not None:
                layer.bias.data[:]*= 0
            layer.weight.data[:]*=0
        else:
            warnings.warn('Unhandled data type, not resetting parameters')

class RecLIFLayerInj(RecLIFLayer):
    NeuronState = namedtuple('NeuronState', ['P', 'Pr', 'Q', 'Qr', 'R', 'Pi','Qi', 'S'])
    def __init__(self, *args, **kwargs):
        super(RecLIFLayerInj, self).__init__(*args, **kwargs)
        self.k=torch.nn.Parameter(1e-4*torch.ones(self.get_out_channels(self.base_layer), dtype=dtype), requires_grad=True)
    
    def init_state(self, Sin_t):
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      Pr=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      Qr=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      R =torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([out_ch] + out_shape).type(dtype).to(device))
        
    def forward(self, Sin_t, Inj=0):
        if self.state is None:
            self.init_state(Sin_t)

        self.k.data[self.k.data<0]=0

        #Forward traces
        state = self.state
        P = self.alpha * state.P + self.tau_m*state.Q
        Q = self.beta  * state.Q + self.tau_s*Sin_t

        #Recurrent traces
        Pr = self.alpha * state.Pr + self.tau_m*state.Qr
        Qr = self.beta  * state.Qr + self.tau_s*state.S

        #Current injection
        Pi = self.alpha * state.Pi + self.tau_m*state.Qi
        Qi = self.beta  * state.Qi + self.tau_s*Inj

        #Refractory
        R = self.alpharp * state.R - state.S * self.wrp

        #Membrane potential
        U = self.base_layer(P) + self.rec_layer(Pr) + R + Pi*(self.k)
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P,
                                      Q=Q,
                                      Pr=Pr,
                                      Qr=Qr,
                                      R= R,
                                      Pi=Pi,
                                      Qi=Qi,
                                      S= S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U


