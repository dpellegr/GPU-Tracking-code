from NAFF import *

def naff(data, n_particles, coord, coord_prime=0, second_half =False):
  tunes=[]
  for i in range (n_particles):
    tune=NAFF_f1(coord(data,i),coord_prime(data,i))
    if (second_half == True):
      tune = 1-tune
    print 'NAFF for particle %i and for %i turns : %f' %(i+1,len(data), tune)
    tunes.append(tune)
  return tunes



