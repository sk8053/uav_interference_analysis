import numpy as np
import scipy.stats
import scipy.constants
import sys;
sys.path.append('..')
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray
from mmwchanmod.sim.chanmod import dir_path_loss_multi_sect
from mmwchanmod.common.constants import LinkState


class UE(object):
    """
    Class for implementing UAV
    """
    def __init__(self, bs_info=None, network = None, channel_info = None, UE_id:int = 0,
                 UAV_Height_max: int = 120, UAV_Height_min:int =10, ground_UE:bool = False
                 , loc:list = [],  drone_antenna_gain = None, enable_3gpp_ch:bool = False):

        self.UAV_Height_MAX = UAV_Height_max # UAV height max
        self.UAV_Height_MIN = UAV_Height_min # UAV height min
        self.network = network
        self.channel_info = channel_info
        thetabw_, phibw_ = 65, 65 # half power of beamwidth
        # velocity setting
        self.ground_UE = ground_UE
        # UAV ID setting
        self.ue_id = UE_id
        frequency = channel_info.frequency

        self.lambda_ = scipy.constants.speed_of_light / frequency
        self.wrx = 0
        self.rx_sv = dict()
        self.ue_elem_gain = dict()
        self.bs_elem_gain = dict()
        self.bs_locations = np.empty(shape=[0, 3])

        # obtain the information on all BSs
        self.cell_types = np.array([])
        self.arr_gnb_list = []

        #self.pl_gain = np.zeros(len(bs_info))
        self.received_power = np.zeros(len(bs_info))
        # UE, BS beamforming vectors
        self.ue_w_bf, self.bs_w_bf = dict(), dict()
        # UE and BS spatial signatures
        self.ue_sv, self.bs_sv = dict(), dict()
        # enable 3gpp channel
        self.enable_3gpp_ch = enable_3gpp_ch
        # associated section index of serving BS
        self.serving_bs_sect_ind = 0
        # map from BS id to associated section index
        self.bs_sect_ind_dict = dict()
        # all the BSs
        self.bs_set = bs_info
        # power control parameters for uplink transmission
        self.alpha = channel_info.alpha
        self.P0 = channel_info.P0
        # transmission power of UE, default value is maximum, 23dBm
        self.tx_power = channel_info.UE_TX_power
        # carrier frequency
        self.frequency = frequency
        # collect some information from all the BSs
        for bs in bs_info:
            self.bs_locations = np.vstack((self.bs_locations, bs.get_location()))  # collect the location information
            self.cell_types = np.append(self.cell_types, bs.bs_type)  # get BS types, terrestrial or aerial
            self.arr_gnb_list.append(bs.get_antenna_array())  # get antenna arrays for all BSs

        ## intial connection with a random BS
        self.serving_bs_id = -1
        # location of UE
        self.loc = loc
        # index of resource
        self.resource_id = -1
        # set initial location randomly
        # UAV antenna element array setting
        element_ue = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
        if self.frequency ==2e9: # and self.ground_UE is True:
            arr_ue = URA(elem=element_ue, nant=np.array([1, 1]), fc=frequency, drone_antenna_gain=drone_antenna_gain)
        elif self.frequency ==140e9:
            arr_ue = URA(elem=element_ue, nant=np.array([8, 8]), fc=frequency, drone_antenna_gain=drone_antenna_gain)
        else:
            arr_ue = URA(elem=element_ue, nant=np.array([4, 4]), fc=frequency, drone_antenna_gain=drone_antenna_gain)

       # else:
        #    raise ValueError("The antenna configuration is not set for the frequency,", self.frequency)
        # configure antenna arrays for UAVs and UEs
        rand_azm = np.random.uniform(low = -180, high = 180, size=(1,))[0]
        if self.ground_UE is True:
            self.arr_ue = RotatedArray(arr_ue, theta0=0, phi0= rand_azm , drone = False)
            self.arr_ue_list = [self.arr_ue]
        else:
            self.arr_ue = RotatedArray(arr_ue, theta0=-90,phi0=rand_azm,  drone = True)
            self.arr_ue_list =[self.arr_ue]

    # some get-functions
    def get_Tx_power(self):
        return self.tx_power
    def get_id(self):
        return self.ue_id
    def get_array(self):
        return self.arr_ue_list
    def get_bf(self):
        return self.ue_w_bf[self.serving_bs_id]
    def get_ue_type(self):
        serv_ue_loc = np.squeeze(self.get_current_location())
        ue_height = serv_ue_loc[2]
        if ue_height > 20:
            return 'uav'
        else:
            return 'g_ue'
    def get_bs_sect_n(self):
        return self.serving_bs_sect_ind
    def get_resource_index(self):
        return self.resource_id

    def set_resource_index(self, resource_id):
        self.resource_id = resource_id


    def set_channel(self, chan_list=None, link_state = None):
        # set ground user channels that obtained from ns3
        # we assume all ground users are static
        '''
        if np.ndim(chan_list) ==0:
            ue_loc = np.repeat(self.get_current_location().reshape(-1,3), len(self.bs_locations), axis = 0)
            self.g_channel = GroundChannel(tx=self.bs_locations, rx=ue_loc)
            data = open(self.g_channel.path+'ray_tracing.txt', 'r')
            self.g_chan_list = self.g_channel.getList(data)
        else:
            self.g_chan_list = list(chan_list)
        '''
        self.channels = chan_list
        self.link_state = link_state

    def get_random_locations(self):
        """
        obtain the random location over the given network area
        """
        X_MAX, Y_MAX = self.network.X_MAX, self.network.Y_MAX
        X_MIN, Y_MIN = self.network.X_MIN, self.network.Y_MIN
        xx = np.random.uniform(low=X_MIN, high=X_MAX, size=1)
        yy = np.random.uniform(low=Y_MIN, high=Y_MAX, size=1)
        trajectory_2D = np.column_stack((xx, yy))
        uav_height = np.array([self.UAV_Height_MAX])
        loc = np.append(trajectory_2D, uav_height[:, None], axis=1)
        return loc

    def set_locations(self,loc= np.array([])):
        # set location of UAV and UE from outside
        if self.ground_UE is True:
            self.UAV_Height_MAX = 1.6

        if len (loc) == 0: # if location is empy, random locations are chosen
            self.loc = self.get_random_locations()
        else:
            self.loc = np.array(loc).reshape(-1,3)

    def association (self, uav_access_bs_t:bool = True):
        '''
        1) compute channels between UEs (or UAVs) and BSs
        2) save all the data related to channel information
        3) perform association based on the received power from BSs
        4) choose optimal one BS and its sector as serving BS and sector
        5) convey to BSs pathloss gain including beamforming and element gains caused by side lobe

        Parameters
        ----------
        uav_access_bs_t: bool
        enable UAVs to access to terrestrial BSs
        Returns
        -------
        None
        '''
        if len(self.loc) == 0:
            self.loc = self.get_random_locations()

        # first, get channels for all BSs
        loc = self.loc.reshape(1,3)
        dist_vectors = loc - self.bs_locations
        dist_vectors = self.wrap_around(dist_vectors)
        '''
        if self.ground_UE is False:
            if self.enable_3gpp_ch is False:
                self.channels, link_states = self.channel_info.channel.sample_path(dist_vectors, self.cell_types)

            else:
                self.channels = self.g_chan_list
        else:
            self.channels = self.g_chan_list
        '''
        dist3D = np.linalg.norm(dist_vectors, axis = 1)
        bs_sect_ind_dict = dict()
        # then compute path loss, channel matrices, and beamforming vectors for all links
        for bs_id, channel in enumerate(self.channels):

            data = dir_path_loss_multi_sect(self.arr_gnb_list[bs_id], self.arr_ue_list, channel, dist3D[bs_id],
                                            long_term_bf=self.channel_info.long_term_bf,
                                            isdrone=not self.ground_UE, frequency=self.frequency, uplink = True)
            if len(channel.pl) ==0: # outage case
                channel.pl = np.array([250])
                #self.pl_gain[bs_id] = np.zeros((1,))  # path gain = path loss + bf gain + elem gain
                self.bs_elem_gain[bs_id] = {i:np.array([-250]) for i in range(3)}
                self.ue_elem_gain[bs_id] = {i:np.array([-250]) for i in range(3)}
                if self.frequency == 2e9: # and self.ground_UE is True:
                    N_ue = 1
                    N_bs = 8
                elif self.frequency ==140e9:
                    N_ue = 64
                    N_bs = 256
                else:
                    N_ue = 16
                    N_bs = 64
                self.ue_w_bf[bs_id] = np.zeros((N_ue,))  # wrx without element gain
                self.bs_w_bf[bs_id] = np.zeros((N_bs,))  # wtx without element gain
                self.ue_sv[bs_id] = {i:np.zeros((1, N_ue)) for i in range(3)}  # rx_sv without element gain
                self.bs_sv[bs_id] = {i:np.zeros((1, N_bs)) for i in range(3)}  # tx_sv without element gain

                data['ue_sv_dict'] = self.ue_sv[bs_id]
                data['bs_sv_dict'] = self.bs_sv[bs_id]

                self.received_power[bs_id] = -np.inf
                bs_sect_ind = 0
            else:
                # for every UE, make maps from BS ID to channel parameters
                self.ue_sv[bs_id] = data['ue_sv_dict'] # rx_sv without element gain
                self.bs_sv[bs_id] = data['bs_sv_dict'] # tx_sv without element gain
                self.bs_elem_gain[bs_id] = data['ue_elem_gain_dict']
                self.ue_elem_gain[bs_id] = data['bs_elem_gain_dict']

                self.ue_w_bf[bs_id] = data['w_ue'] # wrx without element gain
                self.bs_w_bf[bs_id] = data['w_bs'] # wtx without element gain
                bs_sect_ind = data['sect_ind']
                #bf_gain_at_UE= 10 * np.log10(np.abs(self.ue_sv[bs_id][bs_sect_ind].dot(self.ue_w_bf[bs_id])) ** 2 + 1e-20)
                #print (bf_gain_at_UE)
                received_power_db = self.channel_info.BS_TX_power - channel.pl  + self.bs_elem_gain[bs_id][bs_sect_ind] \
                                    + self.ue_elem_gain[bs_id][bs_sect_ind] #+ bf_gain_at_UE
                received_power_lin = 10**(0.1*received_power_db)

                self.received_power[bs_id] = 10*np.log10(received_power_lin.sum())

            # map from bs id to sector id for one UE
            bs_sect_ind_dict[bs_id] = bs_sect_ind
            # BSs have to store the channel information
            self.bs_set[bs_id].set_channels(ue_id=self.ue_id, link_state=channel.link_state, pl=channel.pl, data=data)

        # Do association
        # find serving BSs having maximum received power
        if self.ground_UE is True: # UEs can only be connected with standard BSs
            max_power = np.max(self.received_power[self.cell_types==1])
        # UAV can connect with either standard BSs and rooftop BSs
        elif self.ground_UE is False and len(self.cell_types) !=0:
            if uav_access_bs_t is True:
                max_power = np.max(self.received_power) # [self.cell_types == 0]
            else: # in this case, UAV can only access to dedicated BSs.
                max_power = np.max(self.received_power[self.cell_types == 0])
        else:
            raise RuntimeError('UAVs are trying to connects to dedicated BSs, but there is no dedicated BS ')


        serv_bs_id = np.where(self.received_power == max_power)[0][0]

        self.serving_bs_id = serv_bs_id

        if self.channels[serv_bs_id].link_state != LinkState.no_link:

            self.bs_set[serv_bs_id].connect_ue(self.ue_id, self.received_power[serv_bs_id], self.get_ue_type())
            self.serving_bs_sect_ind = bs_sect_ind_dict[serv_bs_id]

            # convey index of serving sector of BS to serving BS
            self.bs_set[serv_bs_id].set_serv_bs_sect(n = self.serving_bs_sect_ind)

            # compute Tx power of UE according to simple power control algorithm according to the formula given to 3GPP
            if self.channel_info.ptrl is True:
                P_ex = (self.channels[serv_bs_id].pl - self.bs_elem_gain[serv_bs_id][self.serving_bs_sect_ind]
                                - self.ue_elem_gain[serv_bs_id][self.serving_bs_sect_ind]) * self.alpha + self.P0
                P_ex = 10*np.log10(np.sum(10**(0.01*P_ex)))
                self.tx_power = min(P_ex, self.channel_info.UE_TX_power)
            else:
                self.tx_power = 23


            for bs_id  in range (len(self.channels)):
                for bs_sect_ind in range(3):
                    if self.channels[bs_id].pl[0] != 250: # if not outage

                        ue_sv_s = self.ue_sv[bs_id][bs_sect_ind] #.T,
                        ue_w_bf = self.ue_w_bf[serv_bs_id]

                        bf_gain_UE_side_lobe =  10*np.log10(np.abs(ue_sv_s.dot(ue_w_bf))**2 + 1e-20)

                        pl_gain_no_BS_after_assocition = self.channels[bs_id].pl - bf_gain_UE_side_lobe - self.ue_elem_gain[bs_id][bs_sect_ind] \
                                                         - self.bs_elem_gain[bs_id][bs_sect_ind]
                        pl_lin_no_BS = 10**(-0.1*(pl_gain_no_BS_after_assocition))
                        pl_gain_no_BS_dB = -10*np.log10(pl_lin_no_BS.sum())


                    else: # outage case
                        pl_gain_no_BS_dB = 250
                    # if bs is not serving BS
                    if bs_id != serv_bs_id:
                        serv_ue = False
                        self.bs_set[bs_id].set_snr_by_sidelobe(ue_tx_power = self.tx_power,
                                                               ue_id=self.ue_id,
                                                             pl_gain_no_BS_after_association=pl_gain_no_BS_dB,
                                                             sect_n=bs_sect_ind, serv_ue=serv_ue)

                    # if bs is serving BS and its sector is serving sector
                    elif bs_id == serv_bs_id and bs_sect_ind == self.serving_bs_sect_ind:
                        serv_ue = True
                        self.bs_set[bs_id].set_snr_by_sidelobe( ue_tx_power = self.tx_power,
                                                            ue_id=self.ue_id,
                                                             pl_gain_no_BS_after_association=pl_gain_no_BS_dB,
                                                             sect_n=bs_sect_ind, serv_ue=serv_ue)

    def get_interference(self, connected_BS = None):
        '''
        interference for downlink case: BS -> UEs
        which is the sum of received powers at UE side from other BSs except serving BSs

        Parameters
        ----------
        connected_BS: list
        the set of associated BSs

        Returns
        -------
        data: a dictionary
        this includes all the information: SINR, SNR, total interference
        intra-cell interference, inter-cell interference, ue id, serving bs id
        beamforming gain, ue type, link state
        '''
        inter_itf, intra_itf = 0,0
        serv_bs_id = self.serving_bs_id # serving BS ID

        serv_tx_rx_bf = 0
        #serv_sec_id = self.serving_bs_sect_ind
        # do not consider outage case
        if self.channels[serv_bs_id].link_state != LinkState.no_link:
            # take the beamforming vector toward serving BS at UE side
            w_ue = self.ue_w_bf[serv_bs_id]
            # consider all the associated BSs
            for bs in connected_BS:
                bs_id = bs.get_id()
                if self.channels[bs_id].link_state != LinkState.no_link: # check outage case
                    # take all the connected UEs and UAVs to a BS
                    connected_ue_ids = bs.get_connected_ue_list()
                    for c_ue_id in connected_ue_ids:
                        # take beamforming vector toward connected UEs at BS
                        w_bs = bs.get_mmse_precoders(c_ue_id)
                        # obtain the index of sector of serving BS
                        serv_sec_id_at_bs = bs.get_serv_bs_sect()
                        # get the channel between UAV and the BS that causes interference
                        # we specify the channel including serving sector index of BSs
                        bs_sv = self.bs_sv[bs_id][serv_sec_id_at_bs].T # BS channel between UE and a sector of BS
                        ue_sv = self.ue_sv[bs_id][serv_sec_id_at_bs].T # UE channel between UE and a sector of BS
                        # obtain multi-path channel from BS -> UE
                        H = ue_sv.dot(np.matrix.conjugate(bs_sv.T))
                        n_t, n_r = H.shape
                        # normalize channel
                        H *=np.sqrt(n_t*n_r)/np.linalg.norm(H, 'fro')

                        # compute the beamforming gain
                        # since w_ue is derived from long-term-beamforming scheme
                        # do not take the conjugate to w_ue
                        tx_rx_bf = np.abs(w_ue.dot(H).dot(w_bs))**2
                        pl_lin = 10**(-0.1*self.channels[bs_id].pl)
                        bs_tx_power = self.channel_info.BS_TX_power
                        # compute received power by taking element gain between the UE and a sector of BS
                        rx_p =  bs_tx_power*tx_rx_bf*pl_lin* \
                                10**(0.1*self.bs_elem_gain[bs_id][serv_sec_id_at_bs]) \
                                * 10**(0.1*self.ue_elem_gain[bs_id][serv_sec_id_at_bs])
                        # if it's not serving BS, received power is interference
                        # inter-cell interference
                        if bs_id != serv_bs_id:
                            itf = rx_p
                            # accumulate all interference values in linear-scale
                            inter_itf += itf.sum()
                        # intra-cell interference
                        elif bs_id == serv_bs_id and c_ue_id != self.ue_id:
                            intra_itf += rx_p.sum()
                        # otherwise it's the signal from serving BS
                        else:
                            l_f = 10*np.log10(rx_p.sum())
                            serv_tx_rx_bf = 10*np.log10(tx_rx_bf)

        # then add interference-power to noise power in linear scale
        KT_lin, NF_lin = 10 ** (0.1 * self.channel_info.KT), 10 ** (0.1 * self.channel_info.NF)
        total_itf = intra_itf + inter_itf
        Noise_Interference = KT_lin * NF_lin * self.channel_info.BW + total_itf

        # obtain SINR in dB-scale
        SINR =  self.channel_info.BS_TX_power + l_f - 10 * np.log10(Noise_Interference)
        SNR = self.channel_info.BS_TX_power + l_f - self.channel_info.KT - self.channel_info.NF - 10*np.log10(self.channel_info.BW)
        # covert interference value from linear scale to dB scale
        # offset value is added in case interference is zero
        total_itf_db = 10*np.log10(total_itf+ 1e-20)

        serv_ue_loc = np.squeeze(self.get_current_location())
        ue_height = serv_ue_loc[2]
        if ue_height > 20:
            ue_type = 'uav'
        else:
            ue_type = 'g_ue'
        link_state = self.channels[serv_bs_id].link_state
        return {'SINR':SINR,
                 'SNR':SNR,
                 'total_itf':total_itf_db,
                 'intra_itf':10*np.log10(intra_itf+1e-20),
                 'inter_itf':10*np.log10(inter_itf+1e-20),
                 'link_state':link_state,
                 'ue_id':self.ue_id,
                 'serving_BS_ID':self.serving_bs_id,
                  'bf_gain':serv_tx_rx_bf,
                   'ue_type':ue_type,
                   'l_f':l_f}


    def get_current_location(self):
        return self.loc

    def wrap_around(self, dist_vec):
        """
        wrap around implementation over the given network area
        """
        dist_x, dist_y, dist_z = dist_vec.T
        ind_x = np.where(self.network.X_MAX - np.abs(dist_x) <= np.abs(dist_x))
        ind_y = np.where(self.network.Y_MAX - np.abs(dist_y) <= np.abs(dist_y))
        dist_x[ind_x] = (-1) * np.sign(dist_x[ind_x]) * (self.network.X_MAX - np.abs(dist_x[ind_x]))
        dist_y[ind_y] = (-1) * np.sign(dist_y[ind_y]) * (self.network.Y_MAX - np.abs(dist_y[ind_y]))
        return np.column_stack((dist_x, dist_y, dist_z))


    def get_channels_to_uav(self, uavs):
        """
        Obtain the channel elements  between the UE and UAVs

        Parameters
        ----------
        uavs: uav objects connected with BSs

        Returns
        -------
        ue_sv: dictionary variable
        spatial signature at UE side

        bs_sv: dictionary variable
        spatial signature at BS side

        ue_elem: dictionary variable
        element gains between UE and UAVs at UE side

        bs_elem: dictionary variable
        element gains between UE and UAVs at UAVs side

        channel_dict: dictionary variable
        channel elements between UE snd UAVs

        """
        uav_locations = []

        for uav in uavs:
            uav_locations.append(uav.get_current_location())
        uav_locations = np.array(uav_locations).reshape(-1,3)

        loc = self.loc.reshape(1, 3) # location of UE
        dist_vectors = uav_locations- loc # distance vector between a UE and UAVs
        dist_vectors = self.wrap_around(dist_vectors)

        cell_types = np.repeat([1], len(dist_vectors))

        channels, link_states = self.channel_info.channel.sample_path(dist_vectors, cell_types)
        dist3D = np.linalg.norm(dist_vectors, axis=1)
        # then compute path loss, channel matrices, and beamforming vectors for all links
        ue_sv, bs_sv = dict(), dict()
        ue_elem, bs_elem = dict(), dict()
        channel_dict = dict()

        for i, (uav, channel) in enumerate(zip(uavs, channels)):
            uav_array = uav.get_array()
            uav_id = uav.get_id()

            # obtain channel matrices between the UE and UAVs
            data = dir_path_loss_multi_sect(uav_array, self.arr_ue_list, channel, dist3D[i],
                                            long_term_bf=self.channel_info.long_term_bf,
                                            isdrone=not self.ground_UE, frequency=self.frequency)

            ue_sv[uav_id]= data['ue_sv_dict'][0]  # rx_sv without element gain
            bs_sv[uav_id] = data['bs_sv_dict'][0]  # tx_sv without element gain
            ue_elem[uav_id] = data['ue_elem_gain_dict'][0]
            bs_elem[uav_id] = data['bs_elem_gain_dict'][0]
            channel_dict[uav_id] = channel

        return ue_sv, bs_sv, ue_elem, bs_elem, channel_dict

    def get_interference_from_uav(self, connected_uavs = None):
        """
        Obtain the interference between the UE and UAVs

        Parameters
        ----------
        connected_uavs: uav objects connected with BSs
        Returns
        -------
        itf_dB: interference between UE and UAVs in dB scale

        """
        ue_sv, uav_sv, ue_elem, uav_elem, channels = self.get_channels_to_uav(connected_uavs)
        # do not consider outage case
        serv_bs_id = self.serving_bs_id
        # take the beamforming vector toward serving BS at UE side
        w_ue = self.ue_w_bf[serv_bs_id]
        # consider all the associated BSs
        itf = 0
        for uav in connected_uavs:
            uav_id = uav.get_id()
            if channels[uav_id].link_state != LinkState.no_link: # check outage case

                # take beamforming vector at UAV side
                w_uav = uav.get_bf()
                # get spatial signature at UAV and UE
                uav_sv_s = uav_sv[uav_id].T
                ue_sv_s = ue_sv[uav_id].T
                # obtain multi-path channel
                H = ue_sv_s.dot(np.matrix.conjugate(uav_sv_s.T))
                n_t, n_r = H.shape
                # normalize channel
                H *=np.sqrt(n_t*n_r)/np.linalg.norm(H, 'fro')

                # compute the beamforming gain
                tx_rx_bf = np.abs(np.conj(w_ue).dot(H).dot(w_uav))**2
                pl_lin = 10**(-0.1*channels[uav_id].pl)
                bs_tx_power = self.channel_info.BS_TX_power

                # compute received power by taking element gain between the UE and UAVs
                rx_p =  bs_tx_power*tx_rx_bf*pl_lin* \
                        10**(0.1*uav_elem[uav_id]) \
                        * 10**(0.1* ue_elem[uav_id])
                itf += rx_p.sum()
        itf_dB = 10 * np.log10(itf + 1e-20)

        return itf_dB