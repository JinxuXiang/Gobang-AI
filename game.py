from AI import *
import tkinter as tk
import time

PIECE_SIZE = 10

click_x = 0
click_y = 0


pieces_x = [i for i in range(32, 523, 35)]
pieces_y = [i for i in range(38, 529, 35)]

coor_black = []
coor_white = []
pos_black = []
pos_white = []

person_flag = 1
# black:1 white:0
color = 1
piece_color = 'black'


def pos_to_coor(pos_x, pos_y):
  if pos_x not in range(15) or pos_y not in range(15):
    print('Wrong pos!')
  return pieces_x[pos_x], pieces_y[pos_y]
def coor_to_pos(coor_x, coor_y):
  if coor_x not in pieces_x or coor_y not in pieces_y:
    print('Wrong coor!')
  return pieces_x.index(coor_x), pieces_y.index(coor_y)
def click_to_pos(click_x, click_y):
  if click_x>523+17 or click_x<32-17 or click_y>529+17 or click_y<38-17:
    print('Wrong click!')
    return -1, -1
  dist_x = [abs(x-click_x) for x in pieces_x]
  dist_y = [abs(y-click_y) for y in pieces_y]
  min_x = dist_x.index(min(dist_x))
  min_y = dist_y.index(min(dist_y))
  return min_x, min_y

#右上方的棋子提示（工具）
def showChange(piece_color):
    side_canvas.delete("show_piece")
    side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE,
                        110 + PIECE_SIZE, 25 + PIECE_SIZE,
                        fill = piece_color, tags = ("show_piece"))

#落子
def putPiece(piece_color):
    global coor_black, coor_white
    canvas.create_oval(click_x - PIECE_SIZE, click_y - PIECE_SIZE,
                       click_x + PIECE_SIZE, click_y + PIECE_SIZE, 
                       fill = piece_color, tags = ("piece"))
    if piece_color == "white":
        coor_white.append( (click_x, click_y) )
        if game_win(pos_white):
            var1.set("白棋赢")
            var2.set("游戏结束")
    elif piece_color == "black":
        coor_black.append( (click_x, click_y) )
        if game_win(pos_black):
            var1.set("黑棋赢")
            var2.set("游戏结束")


#事件监听处理
def coorBack(event):  #return coordinates of cursor 返回光标坐标
    global click_x, click_y, pos_black, pos_white, piece_color
    click_x = event.x
    click_y = event.y
    x,y = click_to_pos(click_x, click_y)
    click_x, click_y = pos_to_coor(x,y)
    pos_all = pos_black + pos_white
    total_pieces = 2
    if (x,y) in pos_all:
        pass
    else:
        if color:
            piece_color = 'black'
            pos_black += [(x,y)]
            putPiece(piece_color)
            showChange("white")
            var.set("执白棋")
            computer_move = Maximize(pos_black, pos_white, -math.inf, math.inf, 0, 0, total_pieces)[0]
            pos_white += [computer_move]
            click_x, click_y = pos_to_coor(computer_move[0], computer_move[1])
            piece_color = 'white'
            putPiece(piece_color)
            showChange("black")
            var.set("执黑棋")
        else:
            piece_color = 'white'
            pos_white += [(x,y)]
            putPiece(piece_color)
            showChange("black")
            var.set("执黑棋")
            computer_move = Maximize(pos_black, pos_white, -math.inf, math.inf, 0, 0, total_pieces)[0]
            pos_black += [computer_move]
            click_x, click_y = pos_to_coor(computer_move[0], computer_move[1])
            piece_color = 'black'
            putPiece(piece_color)
            showChange("white")
            var.set("执白棋")
        

def gameReset():
    global color, coor_black, coor_white, piece_color, click_x, click_y, pos_black, pos_white
    canvas.delete("piece")#删除所有棋子
    coor_black = []       #清空黑棋坐标存储器
    coor_white = []       #清空白棋坐标存储器
    pos_black = []        #清空黑棋坐标存储器
    pos_white = []        #清空白棋坐标存储器
    color = color^1       #选定落子颜色
    
    if color:             #如果黑色
        piece_color = 'black'
        var.set("执黑棋")      #还原提示标签
        var1.set("")          #还原输赢提示标签
        var2.set("")          #还原游戏结束提示标签
        showChange("black")   #还原棋子提示图片
        
    else:                 #如果白色
        piece_color = 'white'
        var.set("执黑棋")      #还原提示标签
        var1.set("")          #还原输赢提示标签
        var2.set("")          #还原游戏结束提示标签
        showChange("black")   #还原棋子提示图片
        pos_black = [(7,7)]
        click_x, click_y = pos_to_coor(7,7)
        piece_color = 'black'
        putPiece(piece_color)
        showChange("white")
        var.set("执白棋")


"""窗口主体"""
root = tk.Tk()

root.title("Gobang")
root.geometry("760x560")

"""棋子提示"""
side_canvas = tk.Canvas(root, width = 220, height = 50)
side_canvas.grid(row = 0, column = 1)
side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE,
                        110 + PIECE_SIZE, 25 + PIECE_SIZE,
                        fill = piece_color, tags = ("show_piece") )
"""棋子提示标签"""
var = tk.StringVar()
var.set("执黑棋")
person_label = tk.Label(root, textvariable = var, width = 12, anchor = tk.CENTER, 
                        font = ("Arial", 20) )
person_label.grid(row = 1, column = 1)

"""输赢提示标签"""
var1 = tk.StringVar()
var1.set("")
result_label = tk.Label(root, textvariable = var1, width = 12, height = 4, 
                        anchor = tk.CENTER, fg = "red", font = ("Arial", 25) )
result_label.grid(row = 2, column = 1, rowspan = 2)

"""游戏结束提示标签"""
var2 = tk.StringVar()
var2.set("")
game_label = tk.Label(root, textvariable = var2, width = 12, height = 4, 
                        anchor = tk.CENTER, font = ("Arial", 18) )
game_label.grid(row = 4, column = 1)

"""重置按钮"""
reset_button = tk.Button(root, text = "重新开始", font = 20, 
                          width = 8, command = gameReset)
reset_button.grid(row = 5, column = 1)

"""棋盘绘制"""
#背景
canvas = tk.Canvas(root, bg = "saddlebrown", width = 540, height = 540)
canvas.bind("<Button-1>", coorBack)  #鼠标单击事件绑定
canvas.grid(row = 0, column = 0, rowspan = 6)
#线条
for i in range(15):
    canvas.create_line(32, (35 * i + 38), 522, (35 * i + 38))
    canvas.create_line((35 * i + 32), 38, (35 * i + 32), 528)
#点
point_x = [3, 3, 11, 11, 7]
point_y = [3, 11, 3, 11, 7]
for i in range(5):
    canvas.create_oval(35 * point_x[i] + 28, 35 * point_y[i] + 33, 
                       35 * point_x[i] + 36, 35 * point_y[i] + 41, fill = "black")

#透明棋子（设置透明棋子，方便后面落子的坐标定位到正确的位置）  
for i in pieces_x:
    for j in pieces_y:
        canvas.create_oval(i - PIECE_SIZE, j - PIECE_SIZE,
                           i + PIECE_SIZE, j + PIECE_SIZE,
                           width = 0, tags = (str(i), str(j)))

#数字坐标
for i in range(15):
    label = tk.Label(canvas, text = str(i + 1), fg = "black", bg = "saddlebrown",
                     width = 2, anchor = tk.E)
    label.place(x = 2, y = 35 * i + 28)
#字母坐标
count = 0
for i in range(65, 81):
    label = tk.Label(canvas, text = chr(i), fg = "black", bg = "saddlebrown")
    label.place(x = 35 * count + 25, y = 2)
    count += 1

"""窗口循环"""
root.mainloop()













    
