# othelloLN_10_Human_Net.py

# 개발 계획: (1) player1, player2 를 neural net 을 이용한 MCTS playbot으로 만들기. => (2) data 생성해서, train용 data와 sgf 저장하기
#             => (3) 여러 게임을 하여 batch data 생성하기.


# 현재 Net 끼리 대결하고나서 sgf file 하나를 저장하도록 되어 있음 (현재 net은 0.pkl, 1.pkl 모두 random weight로 된 net임)
# => othello_6_2_Human_Net.py: 한쪽 팀은 입력하여 게임하도록 수정
# => othello_6.py: 이후는 multiple self play로 data도 생성하고 sfg는 한 파일에 collection으로 여러 게임을 저장토록 함.

"""
* othello_6_3 에서 수정된 부분
- policy noise 25%를 검색을 시작 직전에 root node에만 적용하기. (그동안은 모든 node에 적용하였었는데, search 효율이 떨어지는 요인이었을 것 같음)
* othello_6_5 에서 수정된 부분
- MCTS search할 때, 게임이 끝부분에 도달했을 때에는, score를 계산해서 State.V에 넣어 주도록 함.
- MCTS search 횟수를 400회로 낮춤 (<- 700)

* othello_6_5 수정부분
- MCTS 오류 정정: scoring 하는 부분에 dupeBoard 대신에 mainBoard 로 잘못 들어가 있었음.

* othello_6_6 수정부분
- 이곳 저곳 자잘한 수정
==========================
* othelloLN_7_Human_Net.py
- net 의 크기를 증가시키기.
=====================

* othelloLN_10_Human_Net.py
- getNetMove에서 network predict를 하나씩 하지 않고, TN(thread No)(=10)개씩 묶음으로 하도록 함.
- TN 만큼 exploration 하는 중에 이전 thread 가 지나간 경로로는 VL(virtual loss)(=3)을 두기.
-> getNetMove에서 N수가 같은 것이 여러개인 경우, 첫번째 것이 아니라 이중 random으로 고름.
- state를 만들 때, Q 초기 값은 모두 value로, N 초기 값은 legal 에 대해서 모두 1로 하기. mcts_policy 구할 때는 legal action의 N에서 도로 1씩 빼고 계산하기.
- state.SelectAction에서도 puct 가 같은 것이 여러 개인 경우, 이중 random으로 고르게함.

* 2019.1.15 p692.pkl로 대결: Droid zebra는 검색강도 2수까지는 이김 (검색강도 3수 이상은 못 이김) // Reversi는 Professor까지 이김 (Champion은 못 이김)



"""

import pygame, random, sys
from pygame.locals import *
import datetime

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from model.othellonet import *
from common.functions import *

# ======================================================================

WINDOWWIDTH = 800
WINDOWHEIGHT = 1000
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
OLIVE = (128, 128, 0)
TEXTCOLOR = BLACK
BACKGROUNDCOLOR = OLIVE
BOARDCOLOR = GREEN
BOARDLINECOLOR = BLACK
LINECOLOR = BLACK
MESSAGEBOXCOLOR = WHITE
LASTMOVECOLOR = RED
LASTMOVESIZE = 5
STONERADIUS = 22
TABLESTONERADIUS = 15
FILLED = 0
EMPTY = 1
EXPLORATIONCONSTANT = 1
UCTITERTIME = 500

P1NAME = 'Net'
P2NAME = 'Human'

P1Param = 'model/p856.pkl'
#P2Param = 'model/p42.pkl'

P1MCTSITERTIME = 800
#P2MCTSITERTIME = 200

NOISE_POLICY = 0.20
CPUCT = 2

TN = 10
VL = 1

SGF_CONTENT = ''
SGFFORMAT = '(;GM[2]AP[ShimSS]SZ[8]FF[4]GN[Net vs Human]GC[]EV[]RO[]PC[]DT[{0}]TM[]PB[{1}]BR[]PW[{2}]WR[]RE[{3}]\n {4})\n\n'
DT = datetime.datetime.now().isoformat(timespec = 'minutes').replace(':', '_')
PB = None
PW = None
RE = None
MoveSeq = []
MS = None

SGF_NAME = 'sgf/NetVsHuman' + DT + '.sgf'

# =================================================================================


class State:
    def __init__(self, board, tile, parentState, priorAction, legalActions, nonlegal_mask, feature_x, policy, value):
        self.board = board
        self.tile = tile

        if self.tile=='X': self.opponent_tile='O'
        else: self.opponent_tile = 'X'
        
        self.parentState = parentState
        self.priorAction = priorAction
        self.childStates = {}

        for x in range(-1, 64):
            self.childStates[x] = None   # -1은 pass인 child state

        self.legalActions = legalActions
        self.nonlegal_mask = nonlegal_mask
        self.feature_x = feature_x
        self.P = policy
        self.V = value
        
        self.Q = np.full([1, 64], value, dtype=float).astype(np.float16)
        self.N = np.ones([1, 64], dtype=int)
        self.N[self.nonlegal_mask] = 0
        
        #self.V_true_scoring = False
        # print('Another state was created!')
        

    def SelectAction(self):
        if self.legalActions == []:
            return -1
        puct = self.Q + CPUCT*self.P*np.sqrt(np.sum(self.N))/(1+self.N)
        puct[self.nonlegal_mask] = -100
        #selectedAction = np.argmax(puct)

        argmax_list = []
        PUCT_largest = -2.0
        for i in range(64):
            PUCT_current = puct[0][i]
            if PUCT_current < PUCT_largest: pass
            elif PUCT_current > PUCT_largest:
                PUCT_largest = PUCT_current
                argmax_list = []
                argmax_list.append(i)
            else:
                argmax_list.append(i)
        random.shuffle(argmax_list)
        selectedAction = argmax_list[0]
        
        assert(self.nonlegal_mask[0][selectedAction] == False)
        return selectedAction

    def AddChild(self, childBoard, childTile, action, legalActions, nonlegal_mask, feature_x, policy, value):
        state = State(childBoard, childTile, self, action, legalActions, nonlegal_mask, feature_x, policy, value)
        self.childStates[action] = state
        return state

    def Update(self, o_reward, x_reward, VL):
        if self.priorAction != -1:
            self.parentState.N[0][self.priorAction] += 1 - VL
            if self.tile == 'O':
                reward = x_reward
            else: reward = o_reward
            self.parentState.Q[0][self.priorAction] += ((VL-1)*self.parentState.Q[0][self.priorAction]+VL+reward)/self.parentState.N[0][self.priorAction]    # parent의 tile은 'X'


def terminate():
    pygame.quit()
    sys.exit()


def drawText(text, font, surface, x, y):
    textobj = font.render(text, 1, TEXTCOLOR)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


def drawBoard(message, sound):
    # This function prints out the board that it was passed. Returns None.

    windowSurface.fill(BACKGROUNDCOLOR)

    drawText('OTHELLO', font2, windowSurface, 300, 40)


    for x in range(8):
        drawText(chr(ord('a')+x), font3, windowSurface, 215+x*50, 105)
        drawText(chr(ord('a')+x), font3, windowSurface, 215+x*50, 555)
    for y in range(8):
        drawText(chr(ord('a')+y), font3, windowSurface, 165, 155+y*50)
        drawText(chr(ord('a')+y), font3, windowSurface, 615, 155+y*50)


    pygame.draw.rect(windowSurface, BOARDCOLOR, (200, 140, 400, 400))


    for x in range(9):
        pygame.draw.line(windowSurface, BOARDLINECOLOR, (200, 140+x*50), (600, 140+x*50), 3)
        pygame.draw.line(windowSurface, BOARDLINECOLOR, (200+x*50, 140), (200+x*50, 540), 3)

    for y in range(8):
        for x in range(8):
            if mainBoard[x][y] == 'X':
                pygame.draw.circle(windowSurface, BLACK, (225+x*50, 165+y*50), STONERADIUS, FILLED)
            elif mainBoard[x][y] == 'O':
                pygame.draw.circle(windowSurface, WHITE, (225+x*50, 165+y*50), STONERADIUS, FILLED)

    if move != []:
        pygame.draw.rect(windowSurface, LASTMOVECOLOR, (225+move[0]*50-round(LASTMOVESIZE/2), 165+move[1]*50-round(LASTMOVESIZE/2), LASTMOVESIZE, LASTMOVESIZE))

    for x in range(4):
        pygame.draw.line(windowSurface, LINECOLOR, (50, 600+x*50), (750, 600+x*50), 2)

    drawText('TURN', font, windowSurface, 70, 615)
    drawText('POINTS', font, windowSurface, 220, 615)
    drawText('NAME', font, windowSurface, 350, 615)

    pygame.draw.circle(windowSurface, BLACK, (180, 675), TABLESTONERADIUS, FILLED)
    pygame.draw.circle(windowSurface, WHITE, (180, 725), TABLESTONERADIUS, FILLED)

    xscore = 0
    oscore = 0
    for x in range(8):
        for y in range(8):
            if mainBoard[x][y] == 'X':
                xscore += 1
            if mainBoard[x][y] == 'O':
                oscore += 1

    drawText(str(xscore), font, windowSurface, 250, 660)
    drawText(str(oscore), font, windowSurface, 250, 710)

    drawText(xname, font, windowSurface, 350, 660)
    drawText(oname, font, windowSurface, 350, 710)

    if (turn == 'P1' and P1Tile == 'X') or (turn == 'P2' and P2Tile == 'X'): drawText('v', font, windowSurface, 90, 660)
    elif (turn == 'P1' and P1Tile == 'O') or (turn == 'P2' and P2Tile == 'O'): drawText('v', font, windowSurface, 90, 710)

    pygame.draw.rect(windowSurface, MESSAGEBOXCOLOR, (50, 780, 700, 150))
    drawText('MESSAGE', font, windowSurface, 100, 810)
    drawText(message[0], font3, windowSurface, 100, 850)
    drawText(message[1], font3, windowSurface, 100, 880)

    sound.play()
    pygame.display.update()


def resetBoard(board):
    # Blanks out the board it is passed, except for the original starting position.
    for x in range(8):
        for y in range(8):
            board[x][y] = ' '

    # Starting pieces:
    board[3][3] = 'O'
    board[3][4] = 'X'
    board[4][3] = 'X'
    board[4][4] = 'O'


def getNewBoard():
    # Creates a brand new, blank board data structure.
    board = []
    for i in range(8):
        board.append([' '] * 8)

    return board


def isValidMove(board, tile, xstart, ystart):
    # Returns False if the player's move on space xstart, ystart is invalid.
    # If it is a valid move, returns a list of spaces that would become the player's if they made a move here.
    if board[xstart][ystart] != ' ' or not isOnBoard(xstart, ystart):
        return False

    board[xstart][ystart] = tile # temporarily set the tile on the board.

    if tile == 'X':
        otherTile = 'O'
    else:
        otherTile = 'X'

    tilesToFlip = []
    for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
        x, y = xstart, ystart
        x += xdirection # first step in the direction
        y += ydirection # first step in the direction
        if isOnBoard(x, y) and board[x][y] == otherTile:
            # There is a piece belonging to the other player next to our piece.
            x += xdirection
            y += ydirection
            if not isOnBoard(x, y):
                continue
            while board[x][y] == otherTile:
                x += xdirection
                y += ydirection
                if not isOnBoard(x, y): # break out of while loop, then continue in for loop
                    break
            if not isOnBoard(x, y):
                continue
            if board[x][y] == tile:
                # There are pieces to flip over. Go in the reverse direction until we reach the original space, noting all the tiles along the way.
                while True:
                    x -= xdirection
                    y -= ydirection
                    if x == xstart and y == ystart:
                        break
                    tilesToFlip.append([x, y])

    board[xstart][ystart] = ' ' # restore the empty space
    if len(tilesToFlip) == 0: # If no tiles were flipped, this is not a valid move.
        return False
    return tilesToFlip


def isOnBoard(x, y):
    # Returns True if the coordinates are located on the board.
    return x >= 0 and x <= 7 and y >= 0 and y <=7



def getValidMoves(board, tile):
    # Returns a list of [x,y] lists of valid moves for the given player on the given board.
    validMoves = []

    for x in range(8):
        for y in range(8):
            if isValidMove(board, tile, x, y) != False:
                validMoves.append([x, y])
    return validMoves


def getScoreOfBoard(board):
    # Determine the score by counting the tiles. Returns a dictionary with keys 'X' and 'O'.
    xscore = 0
    oscore = 0
    for x in range(8):
        for y in range(8):
            if board[x][y] == 'X':
                xscore += 1
            if board[x][y] == 'O':
                oscore += 1
    return {'X':xscore, 'O':oscore}



def makeMove(board, tile, xstart, ystart):
    # Place the tile on the board at xstart, ystart, and flip any of the opponent's pieces.
    # Returns False if this is an invalid move, True if it is valid.
    tilesToFlip = isValidMove(board, tile, xstart, ystart)

    if tilesToFlip == False:
        return False

    board[xstart][ystart] = tile
    for x, y in tilesToFlip:
        board[x][y] = tile
    return True


def getBoardCopy(board):
    # Make a duplicate of the board list and return the duplicate.
    dupeBoard = getNewBoard()

    for x in range(8):
        for y in range(8):
            dupeBoard[x][y] = board[x][y]

    return dupeBoard


def isOnCorner(x, y):
    # Returns True if the position is in one of the four corners.
    return (x == 0 and y == 0) or (x == 7 and y == 0) or (x == 0 and y == 7) or (x == 7 and y == 7)


def getHumanMove(board, playerTile):
    # Let the player type in their move.
    # Returns the move as [x, y] (or returns the strings 'hints' or 'quit')

    message = ['Enter your move by mouse, or press ESC to exit.', ' ']
    drawBoard(message, clickSound)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()

            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                        terminate()

            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    x = round((event.pos[0]-225)/50)
                    y = round((event.pos[1]-165)/50)

                    if x >= 0 and x <= 7 and y >= 0 and y <= 7:
                        if isValidMove(board, playerTile, x, y) == False:
                            message = ['That is not a valid move!', 'Enter your move by mouse, or press ESC to exit.']
                            drawBoard(message, alertSound)
                        else:
                            return [x, y]


def getComputerMove(board, computerTile):

    message = ['Press any key or click to begin a computer move.', ' ']
    drawBoard(message, clickSound)

    wait_input = True
    while wait_input:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()

            if event.type == KEYDOWN:
                wait_input = False
                    
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                        terminate()
                        
            if event.type == MOUSEBUTTONDOWN:
                if event.button >= 1 and event.button <= 3:
                    wait_input = False

    # Given a board and the computer's tile, determine where to
    # move and return that move as a [x, y] list.
    possibleMoves = getValidMoves(board, computerTile)

    # randomize the order of the possible moves
    random.shuffle(possibleMoves)

    # always go for a corner if available.
    for x, y in possibleMoves:
        if isOnCorner(x, y):
            return [x, y]

    # Go through all the possible moves and remember the best scoring move
    bestScore = -1
    for x, y in possibleMoves:
        dupeBoard = getBoardCopy(board)
        makeMove(dupeBoard, computerTile, x, y)
        score = getScoreOfBoard(dupeBoard)[computerTile]
        if score > bestScore:
            bestMove = [x, y]
            bestScore = score
    return bestMove



def getUCTMove(rootBoard, UCTTile, itermax=UCTITERTIME):

    message = ['Press any key or click to begin a UCT search.', ' ']
    drawBoard(message, clickSound)

    wait_input = True
    while wait_input:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()

            if event.type == KEYDOWN:
                wait_input = False
                    
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                        terminate()
                        
            if event.type == MOUSEBUTTONDOWN:
                if event.button >= 1 and event.button <= 3:
                    wait_input = False

    message = ['UCT is simulating {0} times. Please wait a minute.'.format(UCTITERTIME), ' ']
    drawBoard(message, alertSound)

    rootnode = Node(rootBoard, UCTTile)

    for i in range(itermax):

        node = rootnode
        dupeBoard = getBoardCopy(rootBoard)
        dupeBoardTile = UCTTile
    
        # Select
        while node.untriedMoves == [] and node.childNodes != []: # If Nodes are fully expanded and non-terminal
            node = node.UCTSelectChild()
            if node.priorMove != None:
                makeMove(dupeBoard, dupeBoardTile, node.priorMove[0], node.priorMove[1])
            if dupeBoardTile=='X': dupeBoardTile='O'
            else: dupeBoardTile = 'X'

        # Expand
        if node.untriedMoves == [] and node.childNodes == []: # If Nodes have no legal moves,
            if dupeBoardTile=='X': dupeBoardTile='O'
            else: dupeBoardTile = 'X'

            if getValidMoves(dupeBoard, dupeBoardTile) != []: # while the opponent has legal moves
                node = node.AddChild(dupeBoard, dupeBoardTile, None) # Node: representing a pass
                node.parentNode.childNodes.append(node)

                m = random.choice(node.untriedMoves)
                makeMove(dupeBoard, dupeBoardTile, m[0], m[1])
                if dupeBoardTile=='X': dupeBoardTile='O'
                else: dupeBoardTile = 'X'
                node = node.AddChild(dupeBoard, dupeBoardTile, m)
                node.parentNode.untriedMoves.remove(m)
                node.parentNode.childNodes.append(node)

        elif node.untriedMoves != []: # If Node is non-terminal, expand one more random node.

            m = random.choice(node.untriedMoves)
            makeMove(dupeBoard, dupeBoardTile, m[0], m[1])
            if dupeBoardTile=='X': dupeBoardTile='O'
            else: dupeBoardTile = 'X'
            node = node.AddChild(dupeBoard, dupeBoardTile, m)
            node.parentNode.untriedMoves.remove(m)
            node.parentNode.childNodes.append(node)


        # Rollout
        while True:
            validMoves = getValidMoves(dupeBoard, dupeBoardTile)
            if validMoves == []: 

                if dupeBoardTile=='X': dupeBoardTile='O'
                else: dupeBoardTile = 'X'

                if getValidMoves(dupeBoard, dupeBoardTile) == []:
                    break
                else:
                    continue
                
            else:

                random.shuffle(validMoves)
                notSelected = True

                priority = [[1,8,2,4,4,2,8,1], [8,8,7,7,7,7,8,8], [2,7,3,5,5,3,7,2], [4,7,5,6,6,5,7,4],
                            [4,7,5,6,6,5,7,4], [2,7,3,5,5,3,7,2], [8,8,7,7,7,7,8,8], [1,8,2,4,4,2,8,1]]

                for v in validMoves:
                    if priority[v[0]][v[1]] == 1:
                        m = v
                        notSelected = False
                        break

                if notSelected == True:
                    for v in validMoves:
                        if ((v == [1,0] or v == [0,1] or v == [1,1]) and dupeBoard[0][0] == dupeBoardTile) or ((v == [6,0] or v == [7,1] or v == [6,1]) and dupeBoard[7][0] == dupeBoardTile) or ((v == [0,6] or v == [1,7] or v == [1,6]) and dupeBoard[0][7] == dupeBoardTile) or ((v == [7,6] or v == [6,7] or v == [6,6]) and dupeBoard[0][0] == dupeBoardTile):
                            m = v
                            notSelected = False
                            break

                for i in range(2, 8):
                    if notSelected == True:
                        for v in validMoves:
                            if priority[v[0]][v[1]] == i:
                                m = v
                                notSelected = False
                                break

                if notSelected == True:
                    m = validMoves[0]

                makeMove(dupeBoard, dupeBoardTile, m[0], m[1])
                if dupeBoardTile=='X': dupeBoardTile='O'
                else: dupeBoardTile = 'X'


        score = getScoreOfBoard(dupeBoard)
        if score['O'] > score['X']:
            oscore = 2
            xscore = 0
        elif score['X'] > score['O']:
            oscore = 0
            xscore = 2
        else:
            oscore = 1
            xscore = 1

        # Backpropagate
        while node != None:
            node.Update(oscore, xscore)
            node = node.parentNode

    # Return best move
    for n in rootnode.childNodes:
        print('Move:{0} // wins: {1} // visits: {2} // win rate: {3}'.format(n.priorMove, n.wins, n.visits, n.wins/n.visits))
    print('=====================')
    bestChildNode = sorted(rootnode.childNodes, key = lambda c: c.visits)[-1]
    return bestChildNode.priorMove



def getNetMove(RootState, net, itermax):


    #print('This is the states already present at the start of the MTCS search.')
    #for ac1, st1 in RootState.childStates.items():
    #    if st1 != None:
    #        if ac1 == -1: print('pass')
    #        else:
    #            print(st1.parentState.N[0][ac1])
    #            for ac2, st2 in st1.childStates.items():
    #                if st2 != None:
    #                    if ac2 == -1:
    #                        print('     ', end='')
    #                        print('pass')
    #                    else:
    #                        print('     ', end='')
    #                        print(st2.parentState.N[0][ac2])
    #                        for ac3, st3 in st2.childStates.items():
    #                            if st3 != None:
    #                                if ac3 == -1:
    #                                    print('          ', end='')
    #                                    print('pass')
    #                                else:
    #                                    print('          ', end='')
    #                                    print(st3.parentState.N[0][ac3])
    #                                    for ac4, st4 in st3.childStates.items():
    #                                        if st4 != None:
    #                                            if ac4 == -1:
    #                                                print('               ', end='')
    #                                                print('pass')
    #                                            else:
    #                                                print('               ', end='')
    #                                                print(st4.parentState.N[0][ac4])
    #print('================')


    message = ['Press any key or click to begin a Net MCTS search.', ' ']
    drawBoard(message, clickSound)

    wait_input = True
    while wait_input:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()

            if event.type == KEYDOWN:
                wait_input = False
                    
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                        terminate()
                        
            if event.type == MOUSEBUTTONDOWN:
                if event.button >= 1 and event.button <= 3:
                    wait_input = False

    message = ['Net MCTS is simulating {0} times. Please wait a minute.'.format(itermax), ' ']
    drawBoard(message, alertSound)

    noise = np.random.randn(1, 64).astype(np.float16)
    noise[RootState.nonlegal_mask] = -1000
    noise = softmax(noise)
    noise[RootState.nonlegal_mask] = 0
    RootState.P = RootState.P*(1-NOISE_POLICY) + noise*NOISE_POLICY

    for i in range(int(itermax/TN)):

        threadStack = []
        invert = np.random.randint(2, size=(TN, 3))

        for j in range(TN):

            stackData = []
            
            state = RootState
            dupeBoard = getBoardCopy(state.board)
        
            # Select & Expand
            while True:
                selectedAction = state.SelectAction()

                if selectedAction == -1 and getValidMoves(dupeBoard, state.opponent_tile) == []: # 게임 끝에 도달한 경우 대개는 이곳에서 처리됨.
                    break

                if selectedAction != -1:
                    makeMove(dupeBoard, state.tile, selectedAction%8, int(selectedAction/8))

                if state.childStates[selectedAction] == None:
                    break

                else:
                    state = state.childStates[selectedAction]
                    state.parentState.N[0][selectedAction] += VL
                    state.parentState.Q[0][selectedAction] += -VL * (1+state.parentState.Q[0][selectedAction])/state.parentState.N[0][selectedAction]


            tile = state.opponent_tile
            opponent_tile = state.tile
            
            stackData.append(state)
            stackData.append(dupeBoard)
            stackData.append(tile)
            stackData.append(selectedAction)

            legalActions = getValidMoves(dupeBoard, tile)
            nonlegal_mask = np.ones([1, 64], dtype=bool)
            if legalActions != []:
                for x, y in legalActions:
                    nonlegal_mask[0][x+y*8] = False

            stackData.append(legalActions)
            stackData.append(nonlegal_mask)
            
            feature_x = np.zeros([2, 8, 8], dtype=int)
            
            for y in range(8):
                for x in range(8):
                    if dupeBoard[x][y] == tile:
                        feature_x[0][y][x] = 1
                    elif dupeBoard[x][y] == opponent_tile:
                        feature_x[1][y][x] = 1

            if invert[j][0]: feature_x = np.transpose(feature_x, (0, 2, 1)) # transpose()
            if invert[j][1]: feature_x = np.flip(feature_x, axis=2) # fliplr
            if invert[j][2]: feature_x = np.flip(feature_x, axis=1) # flipud

            stackData.append(feature_x)
            threadStack.append(stackData)


        total_feature_x = np.zeros([TN, 2, 8, 8], dtype=int)
        
        for j in range(TN):
            total_feature_x[j] = threadStack[j][6]

        total_policy, total_value = net.predict(total_feature_x)


        for j in range(TN):

            state = threadStack[j][0]
            policy = total_policy[j].reshape(8,8)
            nonlegal_mask = threadStack[j][5]
            selectedAction = threadStack[j][3]

            if invert[j][2]: policy = np.flipud(policy)
            if invert[j][1]: policy = np.fliplr(policy)
            if invert[j][0]: policy = np.transpose(policy)
            
            policy = policy.reshape(1, 64)
            policy[nonlegal_mask] = -1000
            policy = softmax(policy)
            #noise = np.random.randn(1, 64).astype(np.float16)
            #noise[self.nonlegal_mask] = -1000
            #noise = softmax(noise)
            #policy = policy*(1-NOISE_POLICY) + noise*NOISE_POLICY
            policy[nonlegal_mask] = 0
            value = total_value[j]

            state = state.AddChild(threadStack[j][1], threadStack[j][2], selectedAction, threadStack[j][4], nonlegal_mask, threadStack[j][6], policy, value)
            state.parentState.N[0][selectedAction] += VL
            state.parentState.Q[0][selectedAction] += -VL * (1+state.parentState.Q[0][selectedAction])/state.parentState.N[0][selectedAction]


            # Backpropagate
            if state.tile == 'O':
                o_reward = value
                x_reward = -value
            else:
                o_reward = -value
                x_reward = value
            
            while state != RootState:
                state.Update(o_reward, x_reward, VL)
                state = state.parentState


    # Return best move
    for n in range(64):
        if RootState.N[0][n] != 0:
            print('Move:[{}{}] => N: {} // Q: {}'.format(chr(ord('a')+n%8), chr(ord('a')+int(n/8)), RootState.N[0][n], RootState.Q[0][n]))
    print('---------------------')

    argmax_list = []
    N_largest = 0
    for i in range(64):
        N_current = RootState.N[0][i]
        if N_current < N_largest: pass
        elif N_current > N_largest:
            N_largest = N_current
            argmax_list = []
            argmax_list.append(i)
        else:
            argmax_list.append(i)
    random.shuffle(argmax_list)
    bestAction = argmax_list[0]

    #bestAction = np.argmax(RootState.N)

    print('* Selected move:[{}{}] => N: {} // Q: {}'.format(chr(ord('a')+bestAction%8), chr(ord('a')+int(bestAction/8)), RootState.N[0][bestAction], RootState.Q[0][bestAction]))
    print('--------------------------')
    print('* Probable sequence: ', end='')
    state = RootState
    while state != None and np.sum(state.N) != 0:
        nextProbableAction = np.argmax(state.N)
        if state.childStates[nextProbableAction] == None: break
        print('{}{} - '.format(chr(ord('a')+nextProbableAction%8), chr(ord('a')+int(nextProbableAction/8))), end='')      
        state = state.childStates[nextProbableAction]
    print('\n============================')
    
    return bestAction



# =======================================================================


pygame.init()
windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
pygame.display.set_caption('OTHELLO')
pygame.mouse.set_visible(True)

# set up fonts
font = pygame.font.SysFont(None, 40)
font2 = pygame.font.SysFont(None, 60)
font3 = pygame.font.SysFont(None, 25)

# set up sounds
clickSound = pygame.mixer.Sound('music/move.wav')
alertSound = pygame.mixer.Sound('music/alert.wav')
#pygame.mixer.music.load('music/background_2.mp3')


if P1NAME == 'Net':
    P1net = OthelloNet()
    P1net.load_params(file_name=P1Param)
    P1RootState = None

if P2NAME == 'Net':
    P2net = OthelloNet()
    P2net.load_params(file_name=P2Param)
    P2RootState = None

# -------------------------------------------------

while True:
    # Reset the board and game.
    mainBoard = getNewBoard()
    resetBoard(mainBoard)

    turn = ' '
    xname = ' '
    oname = ' '
    move = []

    DT = datetime.datetime.now().isoformat(timespec = 'minutes').replace(':', '_')
    MoveSeq = []

    #pygame.mixer.music.play(-1, 0.0)

    message = ['New Game! If player 1 want to be Black or White, press B or W. Black goes first.', 'If you want to exit, press ESC key.']
    drawBoard(message, alertSound)

    wait_input = True
    while wait_input:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()

            if event.type == KEYDOWN:
                if event.key == ord('b'):
                    P1Tile, P2Tile = ['X', 'O']
                    xname = 'P1'+P1NAME
                    if P1NAME == 'Net': xname = xname + P1Param
                    oname = 'P2'+P2NAME
                    if P2NAME == 'Net': oname = oname + P2Param
                    turn = 'P1'
                    wait_input = False
                if event.key == ord('w'):
                    P1Tile, P2Tile = ['O', 'X']
                    xname = 'P2'+P2NAME
                    if P2NAME == 'Net': xname = xname + P2Param
                    oname = 'P1'+P1NAME
                    if P1NAME == 'Net': oname = oname + P1Param
                    turn = 'P2'
                    wait_input = False
                    
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                        terminate()


    if P1NAME == 'Net':
        #----------------
        legalActions = getValidMoves(mainBoard, 'X')
        nonlegal_mask = np.ones([1, 64], dtype=bool)
        if legalActions != []:
            for x, y in legalActions:
                nonlegal_mask[0][x+y*8] = False
                
        feature_x = np.zeros([1, 2, 8, 8], dtype=int)
        
        for y in range(8):
            for x in range(8):
                if mainBoard[x][y] == 'X':
                    feature_x[0][0][y][x] = 1
                elif mainBoard[x][y] == 'O':
                    feature_x[0][1][y][x] = 1

        invert_a = random.randint(0, 1)
        invert_b = random.randint(0, 1)
        invert_c = random.randint(0, 1)

        feature_x_i = feature_x
        if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
        if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
        if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

        policy, value = P1net.predict(feature_x_i)

        policy = policy.reshape(8,8)
        if invert_c: policy = np.flipud(policy)
        if invert_b: policy = np.fliplr(policy)
        if invert_a: policy = np.transpose(policy)
        
        policy = policy.reshape(1, 64)
        policy[nonlegal_mask] = -1000
        policy = softmax(policy)
        policy[nonlegal_mask] = 0
        #------------------

        P1RootState = State(mainBoard, 'X', None, None, legalActions, nonlegal_mask, feature_x, policy, value)

    if P2NAME == 'Net':
        #----------------
        legalActions = getValidMoves(mainBoard, 'X')
        nonlegal_mask = np.ones([1, 64], dtype=bool)
        if legalActions != []:
            for x, y in legalActions:
                nonlegal_mask[0][x+y*8] = False
                
        feature_x = np.zeros([1, 2, 8, 8], dtype=int)
        
        for y in range(8):
            for x in range(8):
                if mainBoard[x][y] == 'X':
                    feature_x[0][0][y][x] = 1
                elif mainBoard[x][y] == 'O':
                    feature_x[0][1][y][x] = 1

        invert_a = random.randint(0, 1)
        invert_b = random.randint(0, 1)
        invert_c = random.randint(0, 1)

        feature_x_i = feature_x
        if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
        if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
        if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

        policy, value = P2net.predict(feature_x_i)

        policy = policy.reshape(8,8)
        if invert_c: policy = np.flipud(policy)
        if invert_b: policy = np.fliplr(policy)
        if invert_a: policy = np.transpose(policy)
        
        policy = policy.reshape(1, 64)
        policy[nonlegal_mask] = -1000
        policy = softmax(policy)
        policy[nonlegal_mask] = 0
        #------------------
        P2RootState = State(mainBoard, 'X', None, None, legalActions, nonlegal_mask, feature_x, policy, value)


    PB = xname
    PW = oname
    MoveSeq = []

    message = [' ']
    while True:
        if turn == 'P1':
            # Player 1 (P1)'s turn.

            if P1NAME == 'Net':
                current_action = getNetMove(P1RootState, P1net, P1MCTSITERTIME)
                move = (current_action%8, int(current_action/8))
                assert(P1RootState.tile == P1Tile)
            elif P1NAME == 'Human':
                move = getHumanMove(mainBoard, P1Tile)
                current_action = move[0]+move[1]*8                
            elif P1NAME == 'UCT': move = getUCTMove(mainBoard, P1Tile)
            else: move = getComputerMove(mainBoard, P1Tile)

            makeMove(mainBoard, P1Tile, move[0], move[1])
            
            if P1Tile == 'X':
                move_color = 'B'
            else: move_color = 'W'
            move_sgf = ' ;' + move_color + '[' + chr(ord('a')+move[0]) + chr(ord('a')+move[1]) + ']'
            MoveSeq.append(move_sgf)


            if P1NAME == 'Net':
                if P1RootState.childStates[current_action] == None:
                    #----------------
                    legalActions = getValidMoves(mainBoard, P2Tile)
                    nonlegal_mask = np.ones([1, 64], dtype=bool)
                    if legalActions != []:
                        for x, y in legalActions:
                            nonlegal_mask[0][x+y*8] = False
                            
                    feature_x = np.zeros([1, 2, 8, 8], dtype=int)
                    
                    for y in range(8):
                        for x in range(8):
                            if mainBoard[x][y] == P2Tile:
                                feature_x[0][0][y][x] = 1
                            elif mainBoard[x][y] == P1Tile:
                                feature_x[0][1][y][x] = 1

                    invert_a = random.randint(0, 1)
                    invert_b = random.randint(0, 1)
                    invert_c = random.randint(0, 1)

                    feature_x_i = feature_x
                    if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
                    if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
                    if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

                    policy, value = P1net.predict(feature_x_i)

                    policy = policy.reshape(8,8)
                    if invert_c: policy = np.flipud(policy)
                    if invert_b: policy = np.fliplr(policy)
                    if invert_a: policy = np.transpose(policy)
                    
                    policy = policy.reshape(1, 64)
                    policy[nonlegal_mask] = -1000
                    policy = softmax(policy)
                    policy[nonlegal_mask] = 0
                    #------------------

                    P1RootState = P1RootState.AddChild(mainBoard, P2Tile, current_action, legalActions, nonlegal_mask, feature_x, policy, value)
                    P1RootState.parentState = None
                else:
                    P1RootState = P1RootState.childStates[current_action]
                    P1RootState.parentState = None

            if P2NAME == 'Net':                
                if P2RootState.childStates[current_action] == None:
                    #----------------
                    legalActions = getValidMoves(mainBoard, P2Tile)
                    nonlegal_mask = np.ones([1, 64], dtype=bool)
                    if legalActions != []:
                        for x, y in legalActions:
                            nonlegal_mask[0][x+y*8] = False
                            
                    feature_x = np.zeros([1, 2, 8, 8], dtype=int)
                    
                    for y in range(8):
                        for x in range(8):
                            if mainBoard[x][y] == P2Tile:
                                feature_x[0][0][y][x] = 1
                            elif mainBoard[x][y] == P1Tile:
                                feature_x[0][1][y][x] = 1

                    invert_a = random.randint(0, 1)
                    invert_b = random.randint(0, 1)
                    invert_c = random.randint(0, 1)

                    feature_x_i = feature_x
                    if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
                    if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
                    if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

                    policy, value = P2net.predict(feature_x_i)

                    policy = policy.reshape(8,8)
                    if invert_c: policy = np.flipud(policy)
                    if invert_b: policy = np.fliplr(policy)
                    if invert_a: policy = np.transpose(policy)
                    
                    policy = policy.reshape(1, 64)
                    policy[nonlegal_mask] = -1000
                    policy = softmax(policy)
                    policy[nonlegal_mask] = 0
                    #------------------
                   
                    P2RootState = P2RootState.AddChild(mainBoard, P2Tile, current_action, legalActions, nonlegal_mask, feature_x, policy, value)
                    P2RootState.parentState = None
                else:
                    P2RootState = P2RootState.childStates[current_action]
                    P2RootState.parentState = None


            
            if getValidMoves(mainBoard, P2Tile) == []:
                if getValidMoves(mainBoard, P1Tile) == []:
                    break
                else:
                    message = ['Player 2 has no legal moves and should pass.', 'Press any key or click to continue']
                    drawBoard(message, alertSound)


                    if P1NAME == 'Net':
                        if P1RootState.childStates[-1] == None:
                            #----------------
                            legalActions = getValidMoves(mainBoard, P1Tile)
                            nonlegal_mask = np.ones([1, 64], dtype=bool)
                            if legalActions != []:
                                for x, y in legalActions:
                                    nonlegal_mask[0][x+y*8] = False
                                    
                            feature_x = np.zeros([1, 2, 8, 8], dtype=int)
                            
                            for y in range(8):
                                for x in range(8):
                                    if mainBoard[x][y] == P1Tile:
                                        feature_x[0][0][y][x] = 1
                                    elif mainBoard[x][y] == P2Tile:
                                        feature_x[0][1][y][x] = 1

                            invert_a = random.randint(0, 1)
                            invert_b = random.randint(0, 1)
                            invert_c = random.randint(0, 1)

                            feature_x_i = feature_x
                            if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
                            if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
                            if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

                            policy, value = P1net.predict(feature_x_i)

                            policy = policy.reshape(8,8)
                            if invert_c: policy = np.flipud(policy)
                            if invert_b: policy = np.fliplr(policy)
                            if invert_a: policy = np.transpose(policy)
                            
                            policy = policy.reshape(1, 64)
                            policy[nonlegal_mask] = -1000
                            policy = softmax(policy)
                            policy[nonlegal_mask] = 0
                            #------------------
                            
                            P1RootState = P1RootState.AddChild(mainBoard, P1Tile, -1, legalActions, nonlegal_mask, feature_x, policy, value)
                            P1RootState.parentState = None
                        else:
                            P1RootState = P1RootState.childStates[-1]
                            P1RootState.parentState = None

                    if P2NAME == 'Net':                
                        if P2RootState.childStates[-1] == None:
                            #----------------
                            legalActions = getValidMoves(mainBoard, P1Tile)
                            nonlegal_mask = np.ones([1, 64], dtype=bool)
                            if legalActions != []:
                                for x, y in legalActions:
                                    nonlegal_mask[0][x+y*8] = False
                                    
                            feature_x = np.zeros([1, 2, 8, 8], dtype=int)
                            
                            for y in range(8):
                                for x in range(8):
                                    if mainBoard[x][y] == P1Tile:
                                        feature_x[0][0][y][x] = 1
                                    elif mainBoard[x][y] == P2Tile:
                                        feature_x[0][1][y][x] = 1

                            invert_a = random.randint(0, 1)
                            invert_b = random.randint(0, 1)
                            invert_c = random.randint(0, 1)

                            feature_x_i = feature_x
                            if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
                            if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
                            if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

                            policy, value = P1net.predict(feature_x_i)

                            policy = policy.reshape(8,8)
                            if invert_c: policy = np.flipud(policy)
                            if invert_b: policy = np.fliplr(policy)
                            if invert_a: policy = np.transpose(policy)
                            
                            policy = policy.reshape(1, 64)
                            policy[nonlegal_mask] = -1000
                            policy = softmax(policy)
                            policy[nonlegal_mask] = 0
                            #------------------
                            
                            P2RootState = P2RootState.AddChild(mainBoard, P1Tile, -1, legalActions, nonlegal_mask, feature_x, policy, value)
                            P2RootState.parentState = None
                        else:
                            P2RootState = P2RootState.childStates[-1]
                            P2RootState.parentState = None


                    wait_input = True
                    while wait_input:
                        for event in pygame.event.get():
                            if event.type == QUIT:
                                terminate()

                            if event.type == KEYDOWN:
                                wait_input = False
                                    
                            if event.type == KEYUP:
                                if event.key == K_ESCAPE:
                                        terminate()
                                        
                            if event.type == MOUSEBUTTONDOWN:
                                if event.button >= 1 and event.button <= 3:
                                    wait_input = False

                    
                    continue
            else:
                turn = 'P2'

                        

        else:
            # Player 2 (P2)'s turn.

            if P2NAME == 'Net':
                current_action = getNetMove(P2RootState, P2net, P2MCTSITERTIME)
                move = (current_action%8, int(current_action/8))
                assert(P2RootState.tile == P2Tile)
            elif P2NAME == 'Human':
                move = getHumanMove(mainBoard, P2Tile)
                current_action = move[0]+move[1]*8
            elif P2NAME == 'UCT': move = getUCTMove(mainBoard, P2Tile)
            else: move = getComputerMove(mainBoard, P2Tile)

            makeMove(mainBoard, P2Tile, move[0], move[1])

            if P2Tile == 'X':
                move_color = 'B'
            else: move_color = 'W'
            move_sgf = ' ;' + move_color + '[' + chr(ord('a')+move[0]) + chr(ord('a')+move[1]) + ']'
            MoveSeq.append(move_sgf)


            if P2NAME == 'Net':                
                if P2RootState.childStates[current_action] == None:
                    #----------------
                    legalActions = getValidMoves(mainBoard, P1Tile)
                    nonlegal_mask = np.ones([1, 64], dtype=bool)
                    if legalActions != []:
                        for x, y in legalActions:
                            nonlegal_mask[0][x+y*8] = False
                            
                    feature_x = np.zeros([1, 2, 8, 8], dtype=int)
                    
                    for y in range(8):
                        for x in range(8):
                            if mainBoard[x][y] == P1Tile:
                                feature_x[0][0][y][x] = 1
                            elif mainBoard[x][y] == P2Tile:
                                feature_x[0][1][y][x] = 1

                    invert_a = random.randint(0, 1)
                    invert_b = random.randint(0, 1)
                    invert_c = random.randint(0, 1)

                    feature_x_i = feature_x
                    if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
                    if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
                    if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

                    policy, value = P2net.predict(feature_x_i)

                    policy = policy.reshape(8,8)
                    if invert_c: policy = np.flipud(policy)
                    if invert_b: policy = np.fliplr(policy)
                    if invert_a: policy = np.transpose(policy)
                    
                    policy = policy.reshape(1, 64)
                    policy[nonlegal_mask] = -1000
                    policy = softmax(policy)
                    policy[nonlegal_mask] = 0
                    #------------------
                    
                    P2RootState = P2RootState.AddChild(mainBoard, P1Tile, current_action, legalActions, nonlegal_mask, feature_x, policy, value)
                    P2RootState.parentState = None
                else:
                    P2RootState = P2RootState.childStates[current_action]
                    P2RootState.parentState = None

            if P1NAME == 'Net':
                if P1RootState.childStates[current_action] == None:
                    #----------------
                    legalActions = getValidMoves(mainBoard, P1Tile)
                    nonlegal_mask = np.ones([1, 64], dtype=bool)
                    if legalActions != []:
                        for x, y in legalActions:
                            nonlegal_mask[0][x+y*8] = False
                            
                    feature_x = np.zeros([1, 2, 8, 8], dtype=int)
                    
                    for y in range(8):
                        for x in range(8):
                            if mainBoard[x][y] == P1Tile:
                                feature_x[0][0][y][x] = 1
                            elif mainBoard[x][y] == P2Tile:
                                feature_x[0][1][y][x] = 1

                    invert_a = random.randint(0, 1)
                    invert_b = random.randint(0, 1)
                    invert_c = random.randint(0, 1)

                    feature_x_i = feature_x
                    if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
                    if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
                    if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

                    policy, value = P1net.predict(feature_x_i)

                    policy = policy.reshape(8,8)
                    if invert_c: policy = np.flipud(policy)
                    if invert_b: policy = np.fliplr(policy)
                    if invert_a: policy = np.transpose(policy)
                    
                    policy = policy.reshape(1, 64)
                    policy[nonlegal_mask] = -1000
                    policy = softmax(policy)
                    policy[nonlegal_mask] = 0
                    #------------------
                    
                    P1RootState = P1RootState.AddChild(mainBoard, P1Tile, current_action, legalActions, nonlegal_mask, feature_x, policy, value)
                    P1RootState.parentState = None
                else:
                    P1RootState = P1RootState.childStates[current_action]
                    P1RootState.parentState = None

            
            if getValidMoves(mainBoard, P1Tile) == []:
                if getValidMoves(mainBoard, P2Tile) == []:
                    break
                else:
                    message = ['Player 1 has no legal moves and should pass.', 'Press any key or click to continue']
                    drawBoard(message, alertSound)


                    if P1NAME == 'Net':
                        if P1RootState.childStates[-1] == None:
                            #----------------
                            legalActions = getValidMoves(mainBoard, P2Tile)
                            nonlegal_mask = np.ones([1, 64], dtype=bool)
                            if legalActions != []:
                                for x, y in legalActions:
                                    nonlegal_mask[0][x+y*8] = False
                                    
                            feature_x = np.zeros([1, 2, 8, 8], dtype=int)
                            
                            for y in range(8):
                                for x in range(8):
                                    if mainBoard[x][y] == P2Tile:
                                        feature_x[0][0][y][x] = 1
                                    elif mainBoard[x][y] == P1Tile:
                                        feature_x[0][1][y][x] = 1

                            invert_a = random.randint(0, 1)
                            invert_b = random.randint(0, 1)
                            invert_c = random.randint(0, 1)

                            feature_x_i = feature_x
                            if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
                            if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
                            if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

                            policy, value = P1net.predict(feature_x_i)

                            policy = policy.reshape(8,8)
                            if invert_c: policy = np.flipud(policy)
                            if invert_b: policy = np.fliplr(policy)
                            if invert_a: policy = np.transpose(policy)
                            
                            policy = policy.reshape(1, 64)
                            policy[nonlegal_mask] = -1000
                            policy = softmax(policy)
                            policy[nonlegal_mask] = 0
                            #------------------
                            
                            P1RootState = P1RootState.AddChild(mainBoard, P2Tile, -1, legalActions, nonlegal_mask, feature_x, policy, value)
                            P1RootState.parentState = None
                        else:
                            P1RootState = P1RootState.childStates[-1]
                            P1RootState.parentState = None

                    if P2NAME == 'Net':                
                        if P2RootState.childStates[-1] == None:
                            #----------------
                            legalActions = getValidMoves(mainBoard, P2Tile)
                            nonlegal_mask = np.ones([1, 64], dtype=bool)
                            if legalActions != []:
                                for x, y in legalActions:
                                    nonlegal_mask[0][x+y*8] = False
                                    
                            feature_x = np.zeros([1, 2, 8, 8], dtype=int)
                            
                            for y in range(8):
                                for x in range(8):
                                    if mainBoard[x][y] == P2Tile:
                                        feature_x[0][0][y][x] = 1
                                    elif mainBoard[x][y] == P1Tile:
                                        feature_x[0][1][y][x] = 1

                            invert_a = random.randint(0, 1)
                            invert_b = random.randint(0, 1)
                            invert_c = random.randint(0, 1)

                            feature_x_i = feature_x
                            if invert_a: feature_x_i = np.transpose(feature_x_i, (0, 1, 3, 2)) # transpose()
                            if invert_b: feature_x_i = np.flip(feature_x_i, axis=3) # fliplr
                            if invert_c: feature_x_i = np.flip(feature_x_i, axis=2) # flipud

                            policy, value = P2net.predict(feature_x_i)

                            policy = policy.reshape(8,8)
                            if invert_c: policy = np.flipud(policy)
                            if invert_b: policy = np.fliplr(policy)
                            if invert_a: policy = np.transpose(policy)
                            
                            policy = policy.reshape(1, 64)
                            policy[nonlegal_mask] = -1000
                            policy = softmax(policy)
                            policy[nonlegal_mask] = 0
                            #------------------
                            
                            P2RootState = P2RootState.AddChild(mainBoard, P2Tile, -1, legalActions, nonlegal_mask, feature_x, policy, value)
                            P2RootState.parentState = None
                        else:
                            P2RootState = P2RootState.childStates[-1]
                            P2RootState.parentState = None

                    
                    wait_input = True
                    while wait_input:
                        for event in pygame.event.get():
                            if event.type == QUIT:
                                terminate()

                            if event.type == KEYDOWN:
                                wait_input = False
                                    
                            if event.type == KEYUP:
                                if event.key == K_ESCAPE:
                                        terminate()
                                        
                            if event.type == MOUSEBUTTONDOWN:
                                if event.button >= 1 and event.button <= 3:
                                    wait_input = False
                    
                    continue
            else:
                turn = 'P1'


    # Display the final score.

    score = getScoreOfBoard(mainBoard)
    if score['O'] > score['X']:
        RE = 'W+' + str(score['O']-score['X'])
        message = ['White wins by margin of {}! '.format(score['O']-score['X'])]
    elif score['X'] > score['O']:
        RE = 'B+' + str(score['X']-score['O'])
        message = ['Black wins by margin of {}! '.format(score['X']-score['O'])]
    else:
        RE = 'D'
        message = ['Draw!']

    MS = ''.join(MoveSeq)

    SGF_CONTENT = SGF_CONTENT + SGFFORMAT.format(DT, PB, PW, RE, MS)

    #pygame.mixer.music.stop()

    message.append('Press Y key to continue or ESC key to exit.')
    drawBoard(message, alertSound)
    
    wait_input = True
    while wait_input:
        for event in pygame.event.get():
            if event.type == QUIT:
                with open(SGF_NAME, 'w') as sgf_file:
                    sgf_file.write(SGF_CONTENT)
                print('File stored: {}'.format(SGF_NAME))
                terminate()

            if event.type == KEYDOWN:
                if event.key == ord('y'):
                    wait_input = False
                    
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                        with open(SGF_NAME, 'w') as sgf_file:
                            sgf_file.write(SGF_CONTENT)
                        print('File stored: {}'.format(SGF_NAME))
                        terminate()
                        
