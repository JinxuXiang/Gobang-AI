import math
import numpy as np

ab = [(1,0),(0,1),(1,1),(1,-1)]

def print_piece(pos_black, pos_white, now_piece, color):
  result = np.zeros((15,15))
  for p in pos_black:
    result[14-p[1],p[0]] = 1
  for p in pos_white:
    result[14-p[1],p[0]] = 2
  if now_piece and color == 1:
    result[14-now_piece[1],now_piece[0]] = 3
  if now_piece and color == 0:
    result[14-now_piece[1],now_piece[0]] = 4
  print(result)

def available_pos(pos_black, pos_white, total_pieces):
  pos = pos_black + pos_white
  available = []
  for (x,y) in pos:
    if total_pieces <= 8:
      around = [(x+1,y),(x+1,y+1),(x,y+1),(x-1,y+1),(x-1,y),(x-1,y-1),(x,y-1),(x+1,y-1)]
    else:
      around = [(x+1,y),(x+1,y+1),(x,y+1),(x-1,y+1),(x-1,y),(x-1,y-1),(x,y-1),(x+1,y-1),(x+2,y),(x+2,y+2),(x,y+2),(x-2,y+2),(x-2,y),(x-2,y-2),(x,y-2),(x+2,y-2)]
    for a in around:
      if a not in pos and a not in available:
        if a[0] in range(15) and a[1] in range(15):
          available += [a]
  return available

def cal_cluster2(pos, now_x, now_y, a, b, count, inverse = False):
  if inverse:
    a = a*-1
    b = b*-1
  cluster = []
  while (now_x+a,now_y+b) in pos:
    if inverse:
      cluster = [(now_x+a, now_y+b)] + cluster
    else:
      cluster = cluster + [(now_x+a, now_y+b)]
    count += 1
    now_x = now_x+a
    now_y = now_y+b
  return cluster, count

def cluster_count_direction2(pos, direction):
  # direction:
  #  0: left and right
  #  1: up and down
  #  2: right-up and left-down
  #  3: right_down and left_up
  a,b = ab[direction]
  cluster_final = []

  for (x,y) in pos:
    cluster = [(x,y)]
    count = 1
    new_cluster1,count = cal_cluster2(pos,x,y,a,b,count,False)
    new_cluster2,count = cal_cluster2(pos,x,y,a,b,count,True)
    cluster_final += [(new_cluster2 + cluster + new_cluster1, count, direction, False)]
  
  cluster_final = unique_cluster(cluster_final)

  result = []
  for i in range(len(cluster_final)):
    keep = True
    for j in range(len(cluster_final)):
      fi = cluster_final[i]
      fj = cluster_final[j]
      i_s = fi[0][0]
      i_e = fi[0][fi[1]-1]
      j_s = fj[0][0]
      j_e = fj[0][fj[1]-1]    
      if (i_s[0]-2*a,i_s[1]-2*b) == (j_e[0],j_e[1]):
        result += [(fj[0] + fi[0], fi[1]+fj[1], direction, True)]
        #keep = False
      elif (j_s[0]-2*a,j_s[1]-2*b) == (i_e[0],i_e[1]):
        result += [(fi[0] + fj[0], fj[1]+fi[1], direction, True)]
        #keep = False
    if keep:
      result += [cluster_final[i]]
  
  result = unique_cluster(result)

  return result

def unique_cluster(cluster):
  result = []
  for c in cluster:
    keep = True
    for r in result:
      if c[1] == r[1] and c[2] == r[2]:
        count = 0
        for i in range(len(c[0])):
          if c[0][i] == r[0][i]:
            count += 1
        if count == c[1]:
          keep = False
          break
    if keep:
      result += [c]
  return result
        

def cluster_count(pos):
  result = []
  for i in range(4):
    temp = cluster_count_direction2(pos, i)
    for t in temp:
      if t[1]>=1 and t[3] == False:
        result += [t]
      if t[1]>=3 and t[3] == True:
        result += [t]
  return result

def Maximize(pos_me, pos_opp, alpha, beta, depth, start_time, total_pieces):

  moveset = select_pos(pos_me, pos_opp, total_pieces)
  if len(moveset) == 0 or depth == 2:
    return None, Utility(pos_me, pos_opp)

  max_m = None
  max_utility = -math.inf
  result_m = []
  result_utility = []

  for m in moveset:
    next_pos_me = pos_me[:] + [m]
    child, utility = Minimize(next_pos_me, pos_opp, alpha, beta, depth+1, start_time, total_pieces)
    #print(next_pos_me, pos_opp, utility)
    result_m += [m]
    result_utility += [utility]
    if utility > 5000:
      return m, utility
    if utility > max_utility:
      max_m = m 
      max_utility = utility
    if max_utility >= beta:
      break
    if max_utility > alpha:
      alpha = max_utility

  ru_log = []
  for i in range(len(result_utility)):
    if result_utility[i] >= 0:
      ru_log += [math.log10(result_utility[i]+0.1)]
    else:
      ru_log += [-1*math.log10(-1*result_utility[i]+0.1)]

  ru_log = np.ceil(ru_log)
  ru_max = max(ru_log)
  ru_index = []
  ru_value = []
  for i in range(len(result_utility)):
    if ru_log[i] == ru_max:
      ru_index += [i]
      ru_value += [result_utility[i]]

  ru_value_max = min(ru_value)+0.1
  for i in range(len(ru_value)):
    ru_value[i] = np.exp(ru_value[i]/abs(ru_value_max))

  ru_value_sum = sum(ru_value)
  for i in range(len(ru_value)):
    ru_value[i] /= ru_value_sum

  index = ru_index[list(np.random.multinomial(1, ru_value)).index(1)]

  return result_m[index], result_utility[index]

def Minimize(pos_me, pos_opp, alpha, beta, depth, start_time, total_pieces):

  moveset = select_pos(pos_opp, pos_me, total_pieces)
  if len(moveset) == 0 or depth == 2:
    return None, Utility(pos_me, pos_opp)

  min_m = None
  min_utility = math.inf

  for m in moveset:
    next_pos_opp = pos_opp[:] + [m]
    child, utility = Maximize(pos_me, next_pos_opp, alpha, beta, depth+1, start_time, total_pieces)
    #print(next_pos_opp, pos_me, utility)
    if utility < -5000:
      return m, utility
    if utility < min_utility:
      min_m = m
      min_utility = utility
    if min_utility <= alpha:
      break
    if min_utility < beta:
      beta = min_utility

  return min_m, min_utility

def in_board(pos):
  if pos[0]>14 or pos[0]<0 or pos[1]>14 or pos[1]<0:
    return False
  return True

def Utility(pos_me, pos_opp):
  result = 0
  result += score_me(pos_me, pos_opp)
  result -= score_me(pos_opp, pos_me)
  return result

def score_me(pos_me, pos_opp):
  cluster_me = cluster_count(pos_me)
  score = 0
  num_end = 0
  for c in cluster_me:
    piece = c[0]
    a,b = ab[c[2]]
    l = c[1]
    if c[3]:
      now_l = 0
      break_l = False
      for p in range(l-1):
        now_l += 1
        if (piece[p][0]+a, piece[p][1]+b) != piece[p+1]:
          if (piece[p][0]+a,piece[p][1]+b) in pos_opp:
            rest_l1 = now_l-1
            rest_l2 = l-now_l-1
            break_l = True
          else:
            break
      if break_l:
        if (piece[0][0]-a,piece[0][1]-b) in pos_opp or not in_board((piece[0][0]-a,piece[0][1]-b)):
          rest_l1 -= 1
        if (piece[l-1][0]+a,piece[l-1][1]+b) in pos_opp or not in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
          rest_l2 -= 1
        if rest_l1 >= 2:
          score += 10**rest_l1
        if rest_l2 >= 2:
          score += 10**rest_l2
      else:
        if l != 4:
          if ((piece[0][0]-a,piece[0][1]-b) in pos_opp or not in_board((piece[0][0]-a,piece[0][1]-b))) and ((piece[l-1][0]+a,piece[l-1][1]+b) in pos_opp or not in_board((piece[l-1][0]+a,piece[l-1][1]+b))):
            pass
          else:
            if (piece[0][0]-a,piece[0][1]-b) in pos_opp or (piece[l-1][0]+a,piece[l-1][1]+b) in pos_opp or not in_board((piece[0][0]-a,piece[0][1]-b)) or not in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
              l -= 1
              score += 10**l
            else:
              score += 10**l
              if l >= 3:
                num_end += 1
        else:
          score += 10**l
          num_end += 1
        
    else:
      if l == 5:
        return 10**7
      if (piece[0][0]-a,piece[0][1]-b) in pos_opp or (piece[l-1][0]+a,piece[l-1][1]+b) in pos_opp or not in_board((piece[0][0]-a,piece[0][1]-b)) or not in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
        l -= 1
      if l >= 2:
        score += 10**l
      if l >= 3:
        num_end += 1
  if num_end >= 2:
    score += 10**4/2 

  return score

def select_pos(pos_me, pos_opp, total_pieces):
  cluster_me = cluster_count(pos_me)
  possible_point = []
  possible_level = []
  for c in cluster_me:
    piece = c[0]
    a,b = ab[c[2]]
    l = c[1]
    if l >= 4:
      if c[3]:
        new_l = 0
        for p in range(l-1):
          new_l += 1
          if (piece[p][0]+a, piece[p][1]+b) != piece[p+1]:
            if (piece[p][0]+a,piece[p][1]+b) not in pos_opp:
              return [(piece[p][0]+a,piece[p][1]+b)]
            else:
              if new_l >= 3:
                if (piece[0][0]-a,piece[0][1]-b) not in pos_opp and in_board((piece[0][0]-a,piece[0][1]-b)):
                  possible_point += [(piece[0][0]-a,piece[0][1]-b)]
                  possible_level += [1]
              if l-new_l >= 3:
                if (piece[l-1][0]-a,piece[l-1][1]-b) not in pos_opp and in_board((piece[l-1][0]-a,piece[l-1][1]-b)):
                  possible_point += [(piece[l-1][0]+a,piece[l-1][1]+b)]
                  possible_level += [1]
              break
      else:
        if (piece[l-1][0]+a,piece[l-1][1]+b) not in pos_opp and in_board((piece[l-1][0]+a,piece[l-1][1]+b) ):
          return [(piece[l-1][0]+a,piece[l-1][1]+b)]
        if (piece[0][0]-a,piece[0][1]-b) not in pos_opp and in_board((piece[0][0]-a,piece[0][1]-b)):
          return [(piece[0][0]-a,piece[0][1]-b)]
    if l == 3:
      if c[3]:
        for p in range(l-1):
          if (piece[p][0]+a, piece[p][1]+b) != piece[p+1]:
            if (piece[p][0]+a,piece[p][1]+b) not in pos_opp:
              possible_point += [(piece[p][0]+a,piece[p][1]+b)]
              possible_level += [2]
              if in_board((piece[0][0]-a,piece[0][1]-b)):
                possible_point += [(piece[0][0]-a,piece[0][1]-b)]
                possible_level += [1]
              if in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
                possible_point += [(piece[l-1][0]+a,piece[l-1][1]+b)]
                possible_level += [1]
              break
          

      else:
        if (piece[l-1][0]+a,piece[l-1][1]+b) not in pos_opp and (piece[0][0]-a,piece[0][1]-b) not in pos_opp:
          if not in_board((piece[0][0]-a,piece[0][1]-b)) or not in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
            if in_board((piece[0][0]-a,piece[0][1]-b)):
              possible_point += [(piece[0][0]-a,piece[0][1]-b)]
              possible_level += [1]
            if in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
              possible_point += [(piece[l-1][0]+a,piece[l-1][1]+b)]
              possible_level += [1]
          else:
            possible_point += [(piece[0][0]-a,piece[0][1]-b), (piece[l-1][0]+a,piece[l-1][1]+b)]
            possible_level += [2,2]
        else: 
          if (piece[0][0]-a,piece[0][1]-b) not in pos_opp and in_board((piece[0][0]-a,piece[0][1]-b)):
            possible_point += [(piece[0][0]-a,piece[0][1]-b)]
            possible_level += [1]
          if (piece[l-1][0]+a,piece[l-1][1]+b) not in pos_opp and in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
            possible_point += [(piece[l-1][0]+a,piece[l-1][1]+b)]
            possible_level += [1]


  cluster_opp = cluster_count(pos_opp)
  emergency_level = 0
  emergencys = []
  emergency_point = []
  for c in cluster_opp:
    piece = c[0]
    a,b = ab[c[2]]
    l = c[1]
    if l == 3:
      if c[3]:
        for p in range(l-1):
          if (piece[p][0]+a, piece[p][1]+b) != piece[p+1]:
            if (piece[p][0]+a,piece[p][1]+b) in pos_me:
              break
            else:
              if (piece[l-1][0]+a,piece[l-1][1]+b) in pos_me or (piece[0][0]-a,piece[0][1]-b) in pos_me:
                break
              else:
                if in_board((piece[0][0]-a,piece[0][1]-b)) and in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
                  emergency_point += [(piece[l-1][0]+a,piece[l-1][1]+b), (piece[0][0]-a,piece[0][1]-b), (piece[p][0]+a,piece[p][1]+b)]
                  emergencys += [1,1,1]
                break
      else:
        if (piece[l-1][0]+a,piece[l-1][1]+b) in pos_me or (piece[0][0]-a,piece[0][1]-b) in pos_me or not in_board((piece[l-1][0]+a,piece[l-1][1]+b)) or not in_board((piece[0][0]-a,piece[0][1]-b)):
          pass
        else:
          emergency_point += [(piece[l-1][0]+a,piece[l-1][1]+b), (piece[0][0]-a,piece[0][1]-b)]
          emergencys += [1,1]
    if l == 4:
      if c[3]:
        for p in range(l-1):
          if (piece[p][0]+a, piece[p][1]+b) != piece[p+1]:
            if (piece[p][0]+a,piece[p][1]+b) in pos_me:
              break
            else:
              emergency_point += [(piece[p][0]+a,piece[p][1]+b)]
              emergencys += [2]
      else:
        in_board1 = False
        in_board2 = False
        in_pos1 = False
        in_pos2 = False
        if in_board((piece[l-1][0]+a,piece[l-1][1]+b)):
          in_board1 = True
          if (piece[l-1][0]+a,piece[l-1][1]+b) not in pos_me:
            in_pos1 = True
            emergency_point += [(piece[l-1][0]+a,piece[l-1][1]+b)]
            emergencys += [2]
        if in_board((piece[0][0]-a,piece[0][1]-b)):
          in_board2 = True
          if (piece[0][0]-a,piece[0][1]-b) not in pos_me:
            in_pos2 = True
            emergency_point += [(piece[0][0]-a,piece[0][1]-b)]
            emergencys += [2]
        if in_board1 ^ in_board2 and in_pos1 ^ in_pos2:
          emergencys[-1] = [1]
        

  try:
    emergency_level = max(emergencys)
  except:
    emergency_level = 0

  result = []
  if emergency_level == 2:
    for i in range(len(emergencys)):
      if emergencys[i] == 2:
        result += [emergency_point[i]]
  elif emergency_level == 1:
    for i in range(len(emergencys)):
      if emergencys[i] == 1:
        result += [emergency_point[i]]
    for i in range(len(possible_level)):
      if possible_level[i] == 2:
        result += [possible_point[i]]
  else:
    for i in range(len(possible_level)):
      if possible_level[i] == 2:
        result += [possible_point[i]]
  
  if not result:
    result = available_pos(pos_me, pos_opp, total_pieces)


  return result


def game_win(pos):
  cluster = cluster_count(pos)
  for c in cluster:
    if c[1] >= 5 and c[3] == False:
      return True
    if c[1] > 5 and c[3] == True:
      l = c[1]
      piece = c[0]
      now_l = 0
      a,b = ab[c[2]]
      for p in range(l-1):
        now_l += 1
        if (piece[p][0]+a, piece[p][1]+b) != piece[p+1]:
          break
      if now_l >=5 or l-now_l >=5:
        return True
  return False


if __name__=="__main__":
    pos_black = [(7,7)]
    pos_white = []
    pos_all = [pos_black] + [pos_white]
    color = 1
    while True:
      color ^= 1
      if game_win(pos_all[color]):
        break
      
      total_pieces = len(pos_all)
      if color:
        new_piece = Maximize(pos_black, pos_white, -math.inf, math.inf, 0, 0, total_pieces)[0]
        pos_black += [new_piece]
        print('Black: ', new_piece, '\n')
      else:
        new_piece = Maximize(pos_white, pos_black, -math.inf, math.inf, 0, 0, total_pieces)[0]
        pos_white += [new_piece]
        print('White: ', new_piece, '\n')

      print_piece(pos_black, pos_white, new_piece,color)
      print('\n')
