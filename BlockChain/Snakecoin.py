import hashlib as hasher
import datetime as date

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hash_block()

    def hash_block(self):
        sha = hasher.sha256()
        # 因为加密时需要将字符串转化为bytes类型, 3默认编码是 utf-8.
        sha.update(str(self.index).encode('utf-8') + str(self.timestamp).encode('utf-8')
                   + str(self.data).encode('utf-8') + str(self.previous_hash).encode('utf-8'))
        return sha.hexdigest()


def create_genesis_block():
    return Block(0, date.datetime.now(), "Genesis Block", "0")

def create_next_block(last_block):
    this_index = last_block.index + 1
    this_timestamp = date.datetime.now()
    this_data = "I am Block: " + str(this_index)
    this_hash = last_block.hash
    return Block(this_index, this_timestamp, this_data, this_hash)

if __name__ == '__main__':
    block_chain = [create_genesis_block()]
    previous_block = block_chain[0]

    BLOCK_NUM = 20

    for i in range(0, BLOCK_NUM):
        block_to_add = create_next_block(previous_block)
        block_chain.append(block_to_add)
        previous_block = block_to_add
        # Tell everyone about it!
        print("Block #{} has been added to the blockchain!".format(block_to_add.index))
        print("Hash: {}\n".format(block_to_add.hash))


'''
python3中,在 num<128 的时候，使用 chr(num).encode('utf-8') 得到的是 一个 字符的ascii十六进制,而 num>128 的时候，
使用 chr(num).encode('utf-8') 得到的是 两个 字节的ascii十六进制.
'''