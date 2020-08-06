class Battlefield:
    def __init__(self, team1, team2, team1_roe, team2_roe):
        self.__team1 = team1
        self.__team2 = team2
        self.__team1_roe = team1_roe
        self.__team2_roe = team2_roe


    def commence(self):
        while self.__team1.soldier_list and self.__team2.soldier_list:
            self.__team1_roe.aim(self.__team2)
            self.__team2_roe.aim(self.__team1)

            self.__team1.fire(self.__team2)
            self.__team2.fire(self.__team1)

            self.__team1.count_KIA()
            self.__team2.count_KIA()


    def who_win(self):
        result = len(self.__team1.soldier_list) - len(self.__team2.soldier_list)
        if result > 0:
            return -1 # team1 win
        elif result < 0:
            return 1 # team2 win
        else:
            return 0 # draw
