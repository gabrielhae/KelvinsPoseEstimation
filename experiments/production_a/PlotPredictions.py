from pyquaternion import Quaternion
from plotly.graph_objs import Mesh3d
import numpy as np

from plotly import tools
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.figure_factory as ff

class PlotinPLotly:
    
    def numpy_norm_penalty(_, q_pred):
        norm = np.linalg.norm(q_pred,2,axis=-1)
        penalty = np.square(1.0 - norm)
        return penalty    
    
    def send_to_plotly_results(train_labels,validation_labels,predictions):
        q_train=[]
        r=[]
        name=[]
        for image_ann in train_labels:
            q_train.append(image_ann['q_vbs2tango'])
            r.append(image_ann['r_Vo2To_vbs_true'])
            name.append('Train')
            
        train_positionvectors_to_tango = np.array(r)
        train_quaternions_around_tango = np.array(q_train)
        
        q_val=[]
        r=[]
        name=[]
        for image_ann in validation_labels:
            q_val.append(image_ann['q_vbs2tango'])
            r.append(image_ann['r_Vo2To_vbs_true'])
            name.append('Validation')
        
        val_positionvectors_to_tango = np.array(r)
        val_quaternions_around_tango = np.array(q_val)
        
        
        predicted_val_quaternions_around_tango=np.array(predictions[0][:,:])
        predicted_val_positionvectors_to_tango=np.array(predictions[1][:,:])
        
        norm_penalty=PlotinPLotly.numpy_norm_penalty(predicted_val_quaternions_around_tango,predicted_val_quaternions_around_tango)
        
        print("Max Norm Penalyty: " + str(max(norm_penalty)))
        print("Min Norm Penalyty: " + str(min(norm_penalty)))
        
        
        hist_data = [train_positionvectors_to_tango[:,2], val_positionvectors_to_tango[:,2], predicted_val_positionvectors_to_tango[:,2]]
        group_labels = ['Training', 'Validation','Test Val']
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .1,.1])
        plot(fig, filename='compare-z.html')
        
        hist_data = [train_positionvectors_to_tango[:,1], val_positionvectors_to_tango[:,1], predicted_val_positionvectors_to_tango[:,1]]
        group_labels = ['Training', 'Validation','Test Val']
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .1,.1])
        plot(fig, filename='compare-y.html')
        
        hist_data = [train_positionvectors_to_tango[:,0], val_positionvectors_to_tango[:,0], predicted_val_positionvectors_to_tango[:,0]]
        group_labels = ['Training', 'Validation','Test Val']
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .1,.1])
        plot(fig, filename='compare-x.html')
        
#        Xcoord = go.Scatter(
#            x=val_positionvectors_to_tango[:,0],
#            y=predicted_val_positionvectors_to_tango[:,0],
#            mode='markers+text',
#            text=['True', 'Predicted'],
#            textposition='bottom center'
#        )
#        Ycoord = go.Scatter(
#            x=val_positionvectors_to_tango[:,1],
#            y=predicted_val_positionvectors_to_tango[:,1],
#            mode='markers+text',
#            text=['True', 'Predicted'],
#            textposition='bottom center'
#        )
#        Zcoord = go.Scatter(
#            x=val_positionvectors_to_tango[:,2],
#            y=predicted_val_positionvectors_to_tango[:,2],
#            mode='markers+text',
#            text=['True', 'Predicted'],
#            textposition='bottom center'
#        )        
#        
#        fig = tools.make_subplots(rows=1, cols=3,subplot_titles=('X', 'Y','Z'))
#        
#        fig.append_trace(Xcoord, 1, 1)
#        fig.append_trace(Ycoord, 1, 2)
#        fig.append_trace(Zcoord, 1, 3)
#        fig['layout']['xaxis1'].update(title='True')
#        fig['layout']['xaxis2'].update(title='True')
#        fig['layout']['xaxis3'].update(title='True')
#        fig['layout']['yaxis1'].update(title='Pred')
#        fig['layout']['yaxis2'].update(title='Pred')
#        fig['layout']['yaxis3'].update(title='Pred')        
#        fig['layout'].update(scene=dict(aspectmode='manual',
#           aspectratio=go.layout.scene.Aspectratio(x=x.ptp(), y=x.ptp()/y.ptp(), z=z.pyp()/x.ptp()*100)))
#        
#        fig['layout'].update(title='Position Vector Ground Truth Comparisen')
#        plot(fig, filename='pv_gt_compare.html')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows = 1, ncols = 3)
        axes[0].scatter(val_positionvectors_to_tango[:,0],predicted_val_positionvectors_to_tango[:,0], s=2)
        axes[0].title.set_text('Xcoord')
        minval=min(np.amin(val_positionvectors_to_tango[:,0]),np.amin(predicted_val_positionvectors_to_tango[:,0]))
        maxval=max(np.amax(val_positionvectors_to_tango[:,0]),np.amax(predicted_val_positionvectors_to_tango[:,0]))
        axes[0].axis([minval, maxval, minval, maxval])
        axes[0].set_xlabel('True')
        axes[0].set_ylabel('Pred')
        axes[0].set_aspect('equal')
        #plt.show()
        
        axes[1].scatter(val_positionvectors_to_tango[:,1],predicted_val_positionvectors_to_tango[:,1], s=2)
        axes[1].title.set_text('Ycoord')
        minval=min(np.amin(val_positionvectors_to_tango[:,1]),np.amin(predicted_val_positionvectors_to_tango[:,1]))
        maxval=max(np.amax(val_positionvectors_to_tango[:,1]),np.amax(predicted_val_positionvectors_to_tango[:,1]))
        axes[1].axis([minval, maxval, minval, maxval])        
        axes[1].set_xlabel('True')
        axes[1].set_ylabel('Pred')
        axes[1].set_aspect('equal')
        #plt.show()      
        
        axes[2].scatter(val_positionvectors_to_tango[:,2],predicted_val_positionvectors_to_tango[:,2], s=2)
        axes[2].title.set_text('Zcoord')
        minval=min(np.amin(val_positionvectors_to_tango[:,2]),np.amin(predicted_val_positionvectors_to_tango[:,2]))
        maxval=max(np.amax(val_positionvectors_to_tango[:,2]),np.amax(predicted_val_positionvectors_to_tango[:,2]))
        axes[2].axis([minval, maxval, minval, maxval])        
        axes[2].set_xlabel('True')
        axes[2].set_ylabel('Pred')
        axes[2].set_aspect('equal')
        
        fig.tight_layout()
        plt.show()                


        
            
        listviewpoints=[]
        for qt in train_quaternions_around_tango:
            q6 = Quaternion(qt)
            
            new_position=q6.rotate([0.0, 0.0, -1.0]) #camera is positioned 1 metre behind tango for imageing
            listviewpoints.append(new_position)
        
        listviewpoints1=np.array(listviewpoints)
        
        listviewpoints=[]
        for qt in val_quaternions_around_tango:
            q6 = Quaternion(qt)
            
            new_position=q6.rotate([0.0, 0.0, -1.0]) #camera is positioned 1 metre behind tango for imageing
            listviewpoints.append(new_position)
        
        listviewpoints2=np.array(listviewpoints)
        
        listviewpoints=[]
        for qt in predicted_val_quaternions_around_tango:
            q6 = Quaternion(qt)
            
            new_position=q6.rotate([0.0, 0.0, -1.0]) #camera is positioned 1 metre behind tango for imageing
            listviewpoints.append(new_position)
        
        listviewpoints3=np.array(listviewpoints)
        
        train = go.Scatter3d(
            x=listviewpoints1[:,0],
            y=listviewpoints1[:,1],
            z=listviewpoints1[:,2],
               
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    color='red',
                    width=0.5
                ),
                opacity=1.0
            ),
                        name='Train'
        )
                
        validation = go.Scatter3d(
            x=listviewpoints2[:,0],
            y=listviewpoints2[:,1],
            z=listviewpoints2[:,2],
               
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    color='blue',
                    width=0.5
                ),
                opacity=1.0
            ),
                        name='Validation'
        )
                
        test_validation = go.Scatter3d(
            x=listviewpoints3[:,0],
            y=listviewpoints3[:,1],
            z=listviewpoints3[:,2],
               
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    color='blue',
                    width=0.5
                ),
                opacity=1.0
            ),
                        name='Test Val'
        )
        
        
        def sphere():
            scale=0.9
            theta=np.linspace(0, 2*np.pi, 50)
            phi=np.linspace(0, np.pi, 50)
            theta, phi=np.meshgrid(theta, phi)
            x=np.cos(theta)*np.sin(phi)*scale
            y=np.sin(theta)*np.sin(phi)*scale
            z=np.cos(phi)*scale
            return x, y ,z
        
        x,y,z=sphere()
        sphere=Mesh3d({
                        'x': x.flatten(), 
                        'y': y.flatten(), 
                        'z': z.flatten(), 
                        'alphahull': 0
        },
                        name='Sphere')
        # to use with Jupyter notebook
        data=[sphere,train,validation,test_validation]
        plot(data,filename='sphere_viewpoints.html')
        
  
        
        
        return
    
    def send_to_plotly_q_results(train_labels,validation_labels,predictions):
        q_train=[]
        name=[]
        for image_ann in train_labels:
            q_train.append(image_ann['q_vbs2tango'])
            name.append('Train')
            
        train_quaternions_around_tango = np.array(q_train)
        
        q_val=[]
        name=[]
        for image_ann in validation_labels:
            q_val.append(image_ann['q_vbs2tango'])
            name.append('Validation')
        
        val_quaternions_around_tango = np.array(q_val)
        
        
        predicted_val_quaternions_around_tango=np.array(predictions[:,:4])

            
        listviewpoints=[]
        for qt in train_quaternions_around_tango:
            q6 = Quaternion(qt)
            
            new_position=q6.rotate([0.0, 0.0, -1.0]) #camera is positioned 1 metre behind tango for imageing
            listviewpoints.append(new_position)
        
        listviewpoints1=np.array(listviewpoints)
        
        listviewpoints=[]
        for qt in val_quaternions_around_tango:
            q6 = Quaternion(qt)
            
            new_position=q6.rotate([0.0, 0.0, -1.0]) #camera is positioned 1 metre behind tango for imageing
            listviewpoints.append(new_position)
        
        listviewpoints2=np.array(listviewpoints)
        
        listviewpoints=[]
        for qt in predicted_val_quaternions_around_tango:
            q6 = Quaternion(qt)
            
            new_position=q6.rotate([0.0, 0.0, -1.0]) #camera is positioned 1 metre behind tango for imageing
            listviewpoints.append(new_position)
        
        listviewpoints3=np.array(listviewpoints)
        
        train = go.Scatter3d(
            x=listviewpoints1[:,0],
            y=listviewpoints1[:,1],
            z=listviewpoints1[:,2],
               
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    color='red',
                    width=0.5
                ),
                opacity=1.0
            ),
                        name='Train'
        )
                
        validation = go.Scatter3d(
            x=listviewpoints2[:,0],
            y=listviewpoints2[:,1],
            z=listviewpoints2[:,2],
               
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    color='blue',
                    width=0.5
                ),
                opacity=1.0
            ),
                        name='Validation'
        )
                
        test_validation = go.Scatter3d(
            x=listviewpoints3[:,0],
            y=listviewpoints3[:,1],
            z=listviewpoints3[:,2],
               
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    color='blue',
                    width=0.5
                ),
                opacity=1.0
            ),
                        name='Test Val'
        )
        
        
        def sphere():
            scale=0.9
            theta=np.linspace(0, 2*np.pi, 50)
            phi=np.linspace(0, np.pi, 50)
            theta, phi=np.meshgrid(theta, phi)
            x=np.cos(theta)*np.sin(phi)*scale
            y=np.sin(theta)*np.sin(phi)*scale
            z=np.cos(phi)*scale
            return x, y ,z
        
        x,y,z=sphere()
        sphere=Mesh3d({
                        'x': x.flatten(), 
                        'y': y.flatten(), 
                        'z': z.flatten(), 
                        'alphahull': 0
        },
                        name='Sphere')
        # to use with Jupyter notebook
        data=[sphere,train,validation,test_validation]
        plot(data,filename='sphere_viewpoints_qonly.html')
        
  
        
        
        return