from ecg.circular import *

#################
# Circular module
#################

def test_to_angle():

    events = ecg.r_peaks + (np.ones(len(ecg.r_peaks)) * 1500)
    ang = to_angles(ecg.r_peaks, events)
_circular(ang)
plot_circular(ang)

x = np.random.normal(np.pi, 0.5, 100)
y = np.random.uniform(0, np.pi*2, 100)
z = np.concatenate([np.random.normal(np.pi/2, 0.5, 50),
                    np.random.normal(np.pi + np.pi/2, 0.5, 50)])
data = pd.DataFrame(data={'x': x, 'y': y, 'z': z}).melt()
plot_circular(data=data, y='value', hue='variable')
