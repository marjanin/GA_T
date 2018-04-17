# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:15:01 2017

@author: Rohit Annigeri
@author: Ali Marjaninejad
Tension model simulation via Gentic Algorithm
"""
from random import randint
from vpython import *
import numpy as np
import pandas as pd
import datetime
from pandas.io.common import EmptyDataError 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from ga_class import GenticAlgorithm

"""
Custom Classes
"""
"""
class " GUI "
Notes: creates gui to control angles for limbs
"""     
class GUI(object):
    """
    function " __init__ "
    Notes: uses model to intialize angles and limits and states
    """ 
    def __init__(self,model):
        self.__angles = model['angles']
        self.__limits = model['limits']
        self.__s_id = []
        self.__b_id = []
        self.__claw = 0 #open
        self.__state = True #running
    """
    function " s_callback "
    Notes: callback function for sliders for angles
    """     
    def s_callback(self,s):
        i = s.id
        u = self.__limits['u'][i]
        l = self.__limits['l'][i]
        self.__angles[i] = self.__s_id[i].value*(u-l) + l
    """
    function " b_callback "
    Notes: button callback function for clicking
    """         
    def b_callback(self,b):
        if b.id == 0:
            if self.__state == True:
                self.__b_id[0].text ='<b>Run</b>' 
            else:
                self.__b_id[0].text ='<b>Pause</b>' 
            self.__state = not self.__state
        else:
            if self.__claw == 0:
                self.__b_id[1].text = '<b>Release</b>'
                self.__claw = 1
            else:
                self.__b_id[1].text = '<b>Grab</b>'
                self.__claw = 0
    """
    function " build "
    Notes: cerates the gui elements (sliders and button)
    """             
    def build(self):
        self.__b_id.append(button(text='<b>Pause</b>',bind=self.b_callback, id=0))
        scene.append_to_caption('\t')
        self.__b_id.append(button(text='<b>Grab</b>',bind=self.b_callback, id=1))
        scene.append_to_caption('\n\n')
        for i in range(len(self.__angles)):
            caption = '\tJoint ' + str(i+1)
            self.__s_id.append( slider(length=250, bind = self.s_callback, id=i))
            scene.append_to_caption(caption + '\n\n')
    """
    function " read_values "
    Notes: get state of model angles and claw state
    """    
    def read_values(self):
        return self.__angles, self.__claw
    """
    function " read_state "
    Notes: get running state
    """ 
    def read_state(self):
        return self.__state
    
class Arm_segment(object):
    def __init__(self, pos, length):
        self.pos = pos
        self.l = length
    """
    Generate a single arm segement (ball+cylinder)
    Needed to be run during model building/initialization
    """
    def generate(self,angle,phi,pos=None):
        new_pos, axis = self.calculate_pos(angle,phi,pos)
        arm = cylinder(pos=self.pos,axis=axis,radius=0.5)
        joint = sphere(pos=self.pos,radius=0.7,color=color.cyan)
        limb = {'arm':arm, 'joint':joint}
        return limb, new_pos
    
    """
    Calulate end points of arm segments
    """
    def calculate_pos(self,angle,phi,pos=None):
        new_pos = vector(0,0,0)
        if pos != None:
            self.pos = pos
        angle = np.radians(angle)
        new_pos.x = self.pos.x + self.l*np.cos(angle)*np.cos(np.radians(phi))
        new_pos.z = self.pos.z + self.l*np.cos(angle)*np.sin(np.radians(phi))
        new_pos.y = self.pos.y + (self.l*np.sin(angle))
        axis = self.l * (new_pos - self.pos).norm() 
        return new_pos, axis

"""
Class "Dataset"
Dataset creation for saving any custom data into a csv file.
"""
class Dataset(object):
    
    def __init__(self,filename,cols=None):
        self.__dataset = None
        self.__cols = cols
        if cols == None:
            self.__cols =  ['Date'] + [str(i) for i in range(75)]
        self.__filename = filename
        try:
            self.__dataset = pd.read_csv(self.__filename)
            self.__cols = self.__dataset.columns
        except (EmptyDataError, FileNotFoundError)  : 
            self.__dataset = pd.DataFrame(columns=self.__cols)   
    
    def update_dataset(self,new_data):
        data_row = dict()
        for i in range(len(new_data)):
            data_row.update({self.__cols[i]: new_data[i]})
        self.__dataset = self.__dataset.append(data_row, ignore_index=True)
    
    def save_dataset(self):
        self.__dataset = self.__dataset.set_index(self.__dataset.columns[0])
        self.__dataset.to_csv(self.__filename)
"""
Setup vpython scene, camera position
create axes and colors
"""
def create_scene(height=None):
    if height==None:
        scene.height = 600
    else:
        scene.height = height
    scene.width = 800
    scene.forward = vector(0.052912, -0.738750, -1.836577)
    size = 30
    x_axis = cylinder(pos=vector(0,0,0),axis=vector(size,0,0),radius=0.05, color=color.red)
    L_x_axis = label(pos=x_axis.axis,text='x',xoffset=1,yoffset=1, box=False)
    y_axis = cylinder(pos=vector(0,0,0),axis=vector(0,size,0),radius=0.05, color=color.red)
    L_y_axis = label(pos=y_axis.axis,text='y',xoffset=1,yoffset=1, box=False)
    z_axis = cylinder(pos=vector(0,0,0),axis=vector(0,0,size),radius=0.05, color=color.red)
    L_z_axis = label(pos=z_axis.axis,text='z',xoffset=1,yoffset=1, box=False)
    base = box(pos=vector(0,0,0), length=size, height=0.1, width=size) 
    base.color = color.green
    base.opacity = 0.4
    return
   
"""
Calulates angles with refernce to x-z axis
"""     
def update_angles(angles,limits):
    for i in range(len(angles)):
        if angles[i] < limits['l'][i] :
            angles[i] = limits['l'][i]
        elif angles[i] > limits['u'][i]:
            angles[i] = limits['u'][i]
        else:
            pass
    total_angles = [angles[0],angles[1]]
    for i in range(2,len(angles)):
        total_angles.append(angles[i] + total_angles[i-1])
    return total_angles

"""
Repositions arm segments
"""   
def update(model):
    seg, arm, angles, pos = model['seg'],model['arm'],model['angles'],model['pos']
    limits = model['limits']
    total_angles = update_angles(angles,limits)
    claw_state = model['claw_state']
    joint = model['joint']
    for i in range(len(arm)):
        arm[i].pos = pos[i]
        joint[i].pos = pos[i]
        new_pos, axis = seg[i].calculate_pos(total_angles[i+1],phi=total_angles[0],pos=pos[i])
        arm[i].axis = axis
        pos[i+1] = new_pos
    joint[-1].pos = new_pos
    if claw_state == 1:
        joint[-1].color = color.red
    else:
        joint[-1].color = color.blue
    return pos[-1]    

"""
Build/Initialize the arm
Creates a dictionary with arm properties
length, limits, current position of arm segments
current angles, claw/grabber state
"""
       
def init_model(lengths=None, limits=None):
    if lengths == None:
        lengths = [3,6,6]
    if limits == None:
        limits = {'l':[0,0,0],'u':[1,180,180]}
    angles = [0,45,30]
    total_angles = update_angles(angles,limits)
    """
    Base joint creations
    """
    a = cylinder(pos=vector(0,0,0),axis=vector(0,1,0),length=lengths[0],radius=0.5,color=color.white)
    a1_joint = sphere(pos=vector(0,0,0),radius=0.7,color=color.cyan)  
    a.visible = False
    a1_joint.visible = False
    pos=[a.axis]
    seg = []
    arm = []
    joint = []
    j = 0
    for i in range(1,len(total_angles)):
        seg.append(Arm_segment(pos=pos[j],length=lengths[i]))
        limb, new_pos = seg[j].generate(total_angles[i],phi=total_angles[0])
        pos.append(new_pos)
        arm.append(limb['arm'])
        joint.append(limb['joint'])
        j+=1
    joint.append(sphere(pos=pos[-1],radius=0.7,color=color.blue))
    model = {'pos':pos, 'seg':seg, 'arm':arm,'joint':joint,'angles':angles}
    model.update({'lengths':lengths,'limits':limits, 'claw_state':0})
    return model


"""
Class "TargetGenerator"
creates a target visiualization following a specified path
updates position based on current iteration value
"""
class TargetGenerator(object):
    def __init__(self,model,path='circle'):
        self.__total_l = sum(model['lengths'])
        self.__path = path
        pos = vector(0,0,0)
        self.__target = sphere(pos=pos,radius=0.5,color=color.red)
        self.__target.visible = False
        self.__curr_iter = 0
    
    def __move_target(self):
        # radius of circular path
        # center of the circle
        r = 2
        cx, cy = 0, 9
        angle = np.radians(self.__curr_iter)
        self.__target.pos.x = r*np.cos(angle) + cx
        self.__target.pos.y = r*np.sin(angle) + cy
        self.__target.pos.z = 0 
        self.__curr_iter += 1
        if self.__curr_iter > 360:
            self.__curr_iter = 0
        
    def __random_target(self):
        pos = vector(0,0,0)
        max_range = int(0.8*self.__total_l )
        pos.x = randint(int(0.1*self.__total_l ),max_range)
        pos.y = randint(int(0.1*self.__total_l ),max_range)
        pos.z = 0
        self.__target.pos = pos
        
    def get_target(self):
        if self.__path == 'circle':
            self.__move_target()
        elif self.__path == 'random':
            self.__random_target()
        self.__target.visible = True
        return self.__target

"""
Function "vis"
Note: visualization of model in motion using the tension values loaded from
a previously saved data file.
"""
def vis(filename='plot_data.csv'):
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    gui = GUI(model)
    gui.build()
    plot_data = load_plot(filename)
    excursion=plot_data['excursion']
    R_values=plot_data['R_values']
    
    i = 0
    while(True):
        if(gui.read_state()):
            tension = [excursion[i,0], excursion[i,1]]
            R = np.reshape(R_values[i],(2,2))
            pred_angles = tension_to_angles(tension,R)
        
            model['angles'] = [0] + pred_angles
            model['reach'] = update(model)
            i+= 1
            i = i % 360
        rate(200)
    return


"""
function "analytic_soln"
Notes: Analytic solution to find true angles for limb position
"""
def analytic_soln(model,x,y):
    l0,l1,l2 = model['lengths']
    y = y-l0
    l = np.sqrt((x*x) + (y*y))
    angle1 = np.arctan2(y,x) - np.arccos(((l*l) + (l1*l1) - (l2*l2))/(2*l1*l))
    angle2 =  np.arctan2((y - l1*np.sin(angle1)),(x - l1*np.cos(angle1))) - angle1
    return list(np.degrees([angle1,angle2]))
"""
class "TensionModulator"
Notes: Induces variablility  to R values in the matrix as a function of current
iteration cycle. Currently adds a factor of the current value in a sinusoidal
fashion
""" 
class TensionModulator(object):
    def __init__(self,R_inv,iterations):
        self.__R_angles = np.linspace(0,2*np.pi,iterations) 
        self.__mul_factor = 0.1
        R = np.linalg.inv(R_inv)
        self.__R0 = R[0,0]
        self.__R1 = R[0,1]
        self.__R2 = R[1,0]
        self.__R3 = R[1,1]
        self.__R4 = 3
        self.__R5 = 4
        
    def modulate_R(self,R_inv,i):
        R = np.linalg.inv(R_inv)
        R[0,0] = self.__R0*(1 + (self.__mul_factor*np.sin(self.__R_angles[i])))
        R[0,1] = self.__R1*(1 + (self.__mul_factor*np.sin(self.__R_angles[i])))
        R[1,0] = self.__R2*(1 + (self.__mul_factor*np.cos(self.__R_angles[i])))
        R[1,1] = self.__R3*(1 + (self.__mul_factor*np.cos(self.__R_angles[i])))
        R_inv = np.linalg.inv(R)
        return R_inv
    
    """
    Currently unused but useful for s3 based coffeicent
    variation
    """
    def mod_s3_R(self, R4, R5, i):
        R4 = self.__R4*(1 + (self.__mul_factor*np.sin(self.__R_angles[i])))
        R5 = self.__R5*(1 + (self.__mul_factor*np.sin(self.__R_angles[i])))
        return R4, R5
"""
function "tension_to_angles"
Notes: converts excursion/tension to angles for limb using R matrix
"""  
def tension_to_angles(tension,R):
    pred_angles = np.dot(R,np.array(tension))
    return list(pred_angles)
"""
function "calc_S3"
Notes: uses pred angles to calculate excursion value in 3rd dimension
"""  
def calc_S3(pred_angles,ten_mod=None,i=None):
    #The last column of the complete R(2x3 matrix)
    #Fixed value of 3 and 4 to determine excurions value in 3rd dimension 
    value0, value1 = 3, 4
#    value0, value1 = ten_mod.mod_s3_R(value0,value1,i)
    s3 = value0*pred_angles[0] + value1*pred_angles[1]
    return [s3]

"""
function "ga_sim"
Notes: runs the genetic algorithm with limb simulation in realtime
"""     
def ga_sim(save_file='plot_data.csv'):
    #intitalizing the visiualization
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    #set total iterations for limb motion
    #set R matrix and intitalize into GA
    TOTAL_ITER = 360
    R = np.linalg.inv(np.array([[-2,0],[0,-1.5]]))
    ga_model = GenticAlgorithm()
    ga_model.set_R(R)
    tension = ga_model.constrained_individual()
    #Modulate R matrix based on total iterations(i.e in time)
    #Create a target to follow circular path
    ten_mod = TensionModulator(R,TOTAL_ITER)
    target_gen = TargetGenerator(model,'circle')
    
    mean_fit_error, min_fit_error = [],[]
    #Data to be saved and plotted
    new_data = []
    
    cols = ['Index','Target Position x','Target Position y','Pred Position x','Pred Position y']
    cols += ['Target Angle q1','Target Angle q2','Pred Angle q1','Pred Angle q2']
    cols += ['Excursion s{}'.format(i) for i in range(1,4)] + ['Time']
    cols += ['R_00','R_01','R_10','R_11']
    data = Dataset(save_file,cols)
    
    i=0    
    while(i < TOTAL_ITER):
        R = ten_mod.modulate_R(R,i)
        ga_model.set_R(R)
        
        target = target_gen.get_target()
        target_angles = analytic_soln(model,target.pos.x,target.pos.y)
        
        start_time = datetime.datetime.now()
        tension = ga_model.run(np.radians(target_angles).tolist(),tension)
        stop_time = datetime.datetime.now()
        
        pred_angles = tension_to_angles(tension,R)
        s3 = calc_S3(pred_angles)
        pred_angles = np.degrees(pred_angles).tolist()
        
        model['angles'] = [0] + pred_angles
        model['reach'] = update(model)
        
        time_diff = (stop_time - start_time).total_seconds()
        error = mag(model['reach']-target.pos)
#        print('target {} prediction {}'.format(np.radians(target_angles),pred_angles))
        print('Iteration {} Time taken = {} Error = {} '.format(i,time_diff,error))
        
        new_data =[i]+[target.pos.x,target.pos.y]+[model['reach'].x,model['reach'].y]
        new_data += target_angles+pred_angles
        new_data += tension+s3+[time_diff]
        R = np.linalg.inv(R)
        R_mod = R.ravel()
        R = np.linalg.inv(R)
        new_data += R_mod.tolist()
        
        curr_mean_fit_error, curr_min_fit_error = ga_model.get_fit_error()
        if(i == 0):
            mean_fit_error, min_fit_error = curr_mean_fit_error, curr_min_fit_error
            
        data.update_dataset(new_data)
        i+=1
        
    fig = plt.figure()
    fig.suptitle('Fitness error')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    x = [i for i in range(len(mean_fit_error))]
    plt.plot(x,mean_fit_error,'.',label='Mean fitness error')
    plt.plot(x,min_fit_error,'.',label='Min fitness error')
    plt.legend()
    plt.savefig('Fitness.pdf',format='pdf')
    plt.show()
    
    data.save_dataset()

"""
function "load_plot"
Notes: saves plotting data into filename and if stats is enabled
then prints plot data statistics
""" 
def load_plot(filename = 'plot_data.csv',stats=False):
    
    target_positions = None
    pred_positions = None
    target_angles = None
    pred_angles = None
    excursion = None
    time_taken = None
    R_values = None
    
    dataset = pd.read_csv(filename)
    target_position_cols = [col for col in dataset.columns if 'Target Position' in col]
    pred_position_cols = [col for col in dataset.columns if 'Pred Position' in col]
    target_angle_cols = [col for col in dataset.columns if 'Target Angle' in col]
    pred_angle_cols = [col for col in dataset.columns if 'Pred Angle' in col]
    excursion_cols = [col for col in dataset.columns if 'Excursion' in col]
    time_cols = [col for col in dataset.columns if 'Time' in col]
    R_cols = [col for col in dataset.columns if 'R_' in col]
    
    if target_position_cols:
        target_positions = dataset[target_position_cols].values
    if pred_position_cols:
        pred_positions = dataset[pred_position_cols].values
    if target_angle_cols:
        target_angles = dataset[target_angle_cols].values
    if pred_angle_cols:
        pred_angles = dataset[pred_angle_cols].values
    if excursion_cols:
        excursion = dataset[excursion_cols].values
    if time_cols:
        time_taken = dataset[time_cols].values
    if R_cols:
        R_values = dataset[R_cols].values
    if stats == True:
        print(dataset.describe())
        print(dataset[time_cols].sum())
        
    plot_data = {'target_positions':target_positions, 'pred_positions':pred_positions,
                 'target_angles':target_angles,'pred_angles':pred_angles,
                 'excursion':excursion,'time_taken':time_taken,'R_values':R_values}
    return plot_data

"""
function "graphing"
Notes: generates graphs of true and predicted positions,
angles, excursion values and excursion matrix values in time.
""" 
def graphing(plot_data):
    
    target_positions=plot_data['target_positions']
    pred_positions=plot_data['pred_positions']
    target_angles=plot_data['target_angles']
    pred_angles=plot_data['pred_angles']
    excursion=plot_data['excursion']
    time_taken=plot_data['time_taken']
    R_values=plot_data['R_values']
    #Plotting graphs
    if target_positions is not None and pred_positions is not None:
        target_positions = np.array(target_positions)
        pred_positions = np.array(pred_positions)
        fig = plt.figure(1)
        fig.suptitle('Positions')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        x1 = target_positions[:,0]
        y1 = target_positions[:,1]
        x2 = pred_positions[:,0]
        y2 = pred_positions[:,1]
        plt.plot(x1,y1,label='Target positions')
        plt.plot(x2,y2,label='Predition positions')
        plt.legend()
        plt.savefig('Position.pdf',format='pdf')
    #target and predicted angles for limb plotting
    if target_angles is not None and pred_angles is not None:
        target_angles = np.array(target_angles)
        pred_angles = np.array(pred_angles)
        fig = plt.figure(2)
        fig.suptitle('Joint Angles')
        plt.xlabel('Angle 1 (q1)')
        plt.ylabel('Angle 2 (q2)')
        x1 = target_angles[:,0]
        y1 = target_angles[:,1]
        x2 = pred_angles[:,0]
        y2 = pred_angles[:,1]
        plt.plot(x1,y1,label='Target angles')
        plt.plot(x2,y2,label='Predition angles')
        plt.legend()
        plt.savefig('Angle.pdf',format='pdf')
    #excursion values plottting
    if excursion is not None:
        excursion = np.array(excursion)
        fig = plt.figure(3)
        ax = fig.gca(projection='3d')
        fig.suptitle('Excursion Values')
        ax.set_xlabel('S1')
        ax.set_ylabel('S2')
        ax.set_zlabel('S3')
        x = excursion[:,0]
        y = excursion[:,1]
        z = excursion[:,2]
        ax.plot(x,y,z)
        ax.view_init(elev=44, azim=-20)
        plt.savefig('Excursion.pdf',format='pdf')
    #Time taken per iteration plotting
    if time_taken is not None:
        time_taken = np.array(time_taken)
        fig = plt.figure(4)
        fig.suptitle('Time taken per Iteration')
        plt.xlabel('Posture')
        plt.ylabel('Time (s)')
        x = [i for i in range(time_taken.shape[0])]
        y = time_taken
        plt.scatter(x,y)
        plt.savefig('Time_dist.pdf',format='pdf')
        fig = plt.figure(5)
        fig.suptitle('Time taken Distribution')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency')
        plt.hist(y,bins=10,color='blue',rwidth=0.8)
        plt.savefig('Time_hist.pdf',format='pdf')
    #R values plotting
    if R_values is not None:
        R_values = np.array(-R_values)
        fig = plt.figure(6)
        fig.suptitle('R values')
        plt.xlabel('Iteration')
        plt.ylabel('R')
        x = [i for i in range(R_values.shape[0])]
        y1 = R_values[:,0]
        y2 = R_values[:,3]
        plt.plot(x,y1,label='R_0')
        plt.plot(x,y2,label='R_1')
        plt.legend()
        plt.savefig('R_values.pdf',format='pdf')
    plt.show()

"""
main function
comment rest and use one function at a time. 
or use all depending on use case
"""
def main():
    ga_sim(save_file = 'plot_data.csv')
    plot_data = load_plot(filename = 'plot_data.csv',stats=True)
    graphing(plot_data)
    vis(filename = 'plot_data.csv')
    pass
    
if __name__ == '__main__':
    main()