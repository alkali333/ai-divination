import hashlib
import random
import time


class RandomGenerator:
    def get_random_number(self, influences: dict, range: tuple):
        """
        :param influences: {'username': 'Peter', 'date': dt.now()}
        :param range: (1:int, 12:int)
        :return: a random integer
        """

        hexes = {
            'a': '11',
            'b': '12',
            'c': '13',
            'd': '14',
            'e': '15',
            'f': '16'}

        rand_value = self.__randomize(influences)
        possibles = list()

        for i in rand_value:
            if i in hexes.keys():
                i = hexes[i]

            integer_from_rand = int(i)
            if range[0] <= integer_from_rand <= range[1]:
                possibles.append(integer_from_rand)

        return random.choice(possibles)

    def __randomize(self, influences: dict):
        influences[str(time.time())] = time.time() * 1000
        m = hashlib.sha256()
        for i in influences:
            m.update(f"{i}".encode())
            m.update(f"{influences[i]}".encode())

        m.digest()
        return m.hexdigest()


Random = RandomGenerator()
