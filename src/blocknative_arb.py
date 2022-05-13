#!/usr/bin/python
import time
import json
from web3 import Web3, HTTPProvider
from helper_functions import get_acceptable_tokens, get_blocks
import traceback
from thegraph_test import *
import sqlite3 as sql
import datetime as dt
import numpy as np
import asyncio
from eth_account.signers.local import LocalAccount
from web3.middleware import construct_sign_and_send_raw_middleware
from flashbots import flashbot
import rlp
import time
from eth_account.account import Account
import hexbytes
from ABIs import WETH_ABI, uniswap_pair_abi, erc_20_abi, \
    my_contract_abi, chiGasToken_abi, my_contract_bytecode
import websockets
from thegraph_test import UniGraph
from abi import BUNDLE_EXECUTOR_ABI

conn_main_db = sql.connect('etherscan_analysis/main_db.db')

w3 = Web3(HTTPProvider("https://mainnet.infura.io/v3/"))  # Mainnet w3
etherscan_key = ''  # Enter your Etherscan API key here
pvt_key = ''  # Enter your private key here
sig_key = ''  # Enter your signature key here
ETH_ACCOUNT_FROM: LocalAccount = Account.from_key(pvt_key)  # Mainnet ETH account
my_contract = w3.eth.contract(address="contract_address", bytecode=my_contract_bytecode)
ETH_ACCOUNT_SIGNATURE: LocalAccount = Account.from_key(sig_key)

w3.middleware_onion.add(construct_sign_and_send_raw_middleware(ETH_ACCOUNT_FROM))
flashbot(w3, ETH_ACCOUNT_SIGNATURE)

WETH_ERC20_TOKEN = w3.toChecksumAddress('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2')
weth_contract = w3.eth.contract(WETH_ERC20_TOKEN, abi=erc_20_abi)
uniswap_address = '0x7a250d5630b4cf539739df2c5dacb4c659f2488d'
sushiswap_address = '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'.lower()

uniswap_contract = w3.eth.contract(w3.toChecksumAddress('0x7a250d5630b4cf539739df2c5dacb4c659f2488d'),
                                   abi=json.load(open('UniswapABI.json', 'r'))['abi'])
sushiswap_contract = w3.eth.contract(w3.toChecksumAddress('0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'),
                                     abi=json.load(open('UniswapABI.json', 'r'))['abi'])

uni = UniGraph()
sushi = UniGraph(dex='sushi')
sushi.get_all_pairs(size_max=3000, first=500, orderby='reserveETH')
uni.get_all_pairs(size_max=5000, first=500, orderby='reserveETH')


# uni.pairs = pd.read_pickle('pairs.pkl')
def prepare_data(uni):
    uni.pairs['token1_hash'] = uni.pairs['token1_hash'].apply(w3.toChecksumAddress)
    uni.pairs['token0_hash'] = uni.pairs['token0_hash'].apply(w3.toChecksumAddress)
    uni.pairs['id'] = uni.pairs['id'].apply(w3.toChecksumAddress)

    d = uni.pairs[['token0_hash', 'token0_symbol']].set_index('token0_hash').to_dict()['token0_symbol']
    d.update(uni.pairs[['token1_hash', 'token1_symbol']].set_index('token1_hash').to_dict()['token1_symbol'])

    empirical_pairs = list(map(w3.toChecksumAddress,
                               list(filter(lambda x: len(x) == 42,
                                           set(
                                               get_acceptable_tokens(sql,
                                                                     '0x000000005736775feb0c8568e7dee77222a26880',
                                                                     '0x000000005736775feb0c8568e7dee77222a26880') +
                                               get_acceptable_tokens(sql,
                                                                     '0xfad95B6089c53A0D1d861eabFaadd8901b0F8533'.lower(),
                                                                     '0xfad95B6089c53A0D1d861eabFaadd8901b0F8533'.lower()) +
                                               get_acceptable_tokens(sql,
                                                                     '0x00000000b7ca7e12dcc72290d1fe47b2ef14c607'.lower(),
                                                                     '0x00000000b7ca7e12dcc72290d1fe47b2ef14c607'.lower()) +
                                               get_acceptable_tokens(sql,
                                                                     '0x00000000003b3cc22af3ae1eac0440bcee416b40',
                                                                     '0x00000000003b3cc22af3ae1eac0440bcee416b40') +
                                               get_acceptable_tokens(sql,
                                                                     '0x000000000000cb53d776774284822b1298ade47f'.lower(),
                                                                     '0x000000000000cb53d776774284822b1298ade47f'.lower()) +
                                               get_acceptable_tokens(sql,
                                                                     '0x0000000099cb7fc48a935bceb9f05bbae54e8987'.lower(),
                                                                     '0x0000000099cb7fc48a935bceb9f05bbae54e8987'.lower())
                                           )))))
    print('Nr. of empirical pairs ', len(empirical_pairs))
    # Check the % deflationary tokens
    c = dict()
    token_supplies = dict()
    t1 = time.time()
    k = c.keys()
    for i, j in uni.pairs[['id', 'token0_hash', 'token1_hash']].iterrows():
        if j['token0_hash'] == WETH_ERC20_TOKEN or j['token1_hash'] == WETH_ERC20_TOKEN:
            token_hash = j['token0_hash' if j['token0_hash'] != WETH_ERC20_TOKEN else 'token1_hash']
            if token_hash not in k and token_hash in empirical_pairs:
                # First is token liquidity and second is WETH liquidity in the pool
                c[token_hash] = [
                    w3.eth.contract(address=token_hash, abi=erc_20_abi).functions.balanceOf(j['id']),
                    w3.eth.contract(address=WETH_ERC20_TOKEN, abi=WETH_ABI).functions.balanceOf(j['id'])]
                token_supplies[token_hash] = w3.eth.contract(address=token_hash,
                                                             abi=erc_20_abi).functions.totalSupply()
    print(time.time() - t1)
    return c, d, token_supplies

c, d, token_supplies = prepare_data(uni)
c_sushi, d_sushi, token_supplies_sushi = prepare_data(sushi)

def build_bundle(nonce: int,
                 to: str,
                 block_number: str,
                 weth_amount_buy: int,
                 weth_amount_sell: int,
                 weth_amount_contract: int,
                 token_amount: int,
                 contract_router,
                 token_address: str,
                 miner_bribe: int):
    block_number = hex(block_number)[2:]
    weth_amount_hex = hex(int(weth_amount_buy))[2:]
    token_amount_hex = hex(token_amount)[2:]
    # Buy tx
    buy = {'value': 0,
           'gas': 150000,
           'gasPrice': 0,
           'nonce': nonce,  # int(w3.eth.getTransactionCount(acct.address)),
           'to': to,
           'data': '0x{buy_byte}' +
                   block_number +
                   '0' * (28 - len(weth_amount_hex)) + weth_amount_hex +
                   '0' * (28 - len(token_amount_hex)) + token_amount_hex +
                   contract_router.address[2:] +
                   "{gas_tokens}"}
    # signed_transaction_buy = ETH_ACCOUNT_FROM.sign_transaction(buy)
    token_amount_hex = hex(token_amount - 1)[2:]
    weth_amount_hex = hex(int(weth_amount_sell))[2:]
    weth_amount_contract = hex(int(weth_amount_contract))[2:]
    sell = {'value': 0,
            'gas': 150000,
            'gasPrice': miner_bribe,
            'nonce': nonce + 1,  # w3.eth.getTransactionCount(acct.address),
            'to': to,
            'data': '0x{sell_byte}' +
                    block_number +
                    '0' * (28 - len(weth_amount_hex)) + weth_amount_hex +
                    '0' * (28 - len(token_amount_hex)) + token_amount_hex +
                    contract_router.address[2:] +
                    token_address[2:] +
                    '0' * (28 - len(weth_amount_contract)) + weth_amount_contract +
                    "{gas_tokens}"}
    # signed_transaction_sell = ETH_ACCOUNT_FROM.sign_transaction(sell)

    return buy, sell


def get_optimal_bundle(block_number_max,
                       block_target,
                       gas_tokens,
                       bin_,
                       expected_tokens,
                       token_supplies,
                       contract_router,
                       params,
                       ETH_ACCOUNT_FROM,
                       weth_contract,
                       t,
                       bribe_pct,
                       ):

    l = []
    tim = time.time()
    target_tx = rlp.encode([int(t[i], 16) if isinstance(t[i], str) and t[i][:2] == '0x' else int(t[i]) for i in
                            ['nonce', 'gasPrice', 'gas', 'to', 'value',
                             'input', 'v', 'r', 's']]).hex()
    r = []
    buy, sell = build_bundle(w3.eth.getTransactionCount(ETH_ACCOUNT_FROM.address),
                             my_contract.address,
                             block_number_max,
                             gas_tokens,
                             int(bin_[0]),
                             int(bin_[1]),
                             weth_contract.functions.balanceOf(my_contract.address).call() -
                             int(bin_[0]) + int(bin_[1]) - 1,
                             int(expected_tokens),
                             contract_router,
                             token_supplies[w3.toChecksumAddress(params['path'][-1])].address,
                             0)
    buy_template = buy['data']
    sell_template = sell['data']
    if contract_router.functions.token0().call() == weth_contract.address:
        buy_sell_dict = {0: ['08', '09'], 1: ['08', '09'], 2: ['08', '09'], 3: ['01', '02']}
    else:
        buy_sell_dict = {0: ['07', '0a'], 1: ['07', '0a'], 2: ['07', '0a'], 3: ['00', '03']}
    print('Pre simulation time: ', time.time() - tim)
    gas_price = int(w3.eth.fee_history(1, 'latest').baseFeePerGas[0])

    print('Simulation took: ', time.time() - tim)
    gas_price = int(w3.eth.fee_history(1, 'latest').baseFeePerGas[-1])
    print('Gas price: ', gas_price)
    profit = bin_[1] - bin_[0]
    first_tx_price = gas_price
    last_tx_price = int((profit * bribe_pct - l[0] * gas_price) / l[1])
    if last_tx_price > first_tx_price:
        print('Potential profit: ', profit)
        print('Base fee: ', sum(l[:2]) * gas_price)
        print('Miner bribe: ', profit * bribe_pct - sum(l[:2]) * gas_price)
        buy['data'] = buy_template.format(buy_byte=buy_sell_dict[0][0], gas_tokens="")
        sell['data'] = sell_template.format(sell_byte=buy_sell_dict[0][1], gas_tokens="")
        buy['gasPrice'] = first_tx_price
        sell['gasPrice'] = last_tx_price
        signed_transaction_buy = ETH_ACCOUNT_FROM.sign_transaction(buy)
        signed_transaction_sell = ETH_ACCOUNT_FROM.sign_transaction(sell)
        bundle = [
            {
                "signed_transaction": signed_transaction_buy.rawTransaction
            },
            {
                "signed_transaction": hexbytes.HexBytes(target_tx)
            },
            {
                "signed_transaction": signed_transaction_sell.rawTransaction,
            },
        ]
        print('Preparation time: ', time.time() - tim)
        result = list(
            map(lambda x: w3.flashbots.send_bundle(bundle, target_block_number=block_target + x), range(1, 3)))

        print('Simulation time: ', time.time() - tim)
        bal_before = weth_contract.functions.balanceOf(my_contract.address).call(block_identifier=block_target - 1)
        bal_after = weth_contract.functions.balanceOf(my_contract.address).call(block_identifier=w3.eth.blockNumber)
        profit = bal_after - bal_before
        print("Balance before", bal_before)
        print("Balance after", bal_after)
        print("Profit: ", profit)
    return l, r


def expected_return_fees(token_pool, weth_pool, value, fee=997, pct=1):
    return (value * fee * token_pool / (weth_pool * 1000 + value * fee)) * pct

def calculate_arb_profit(my_weth, target_token_pool, target_weth_pool, token_pool, weth_pool, div=10 ** 18, fee=997, pct=1):
    tokens = expected_return_fees(token_pool, weth_pool, my_weth, fee, pct)
    my_weth_after = expected_return_fees(target_weth_pool, target_token_pool, tokens, fee, pct)
    profit = my_weth_after - my_weth
    return profit, tokens, my_weth_after

def binary_search_arbitrage(target_eth, target_token_pool, target_weth_pool, token_pool, weth_pool, error_margin, upper=5, lower=0, pct=1):
    tokens = expected_return_fees(target_token_pool, target_weth_pool, target_eth, pct=pct)
    target_token_pool -= tokens
    target_weth_pool += target_eth
    
   
    while upper - lower > error_margin:
        mid = (upper + lower) / 2 
        higher_profit, tokens_higher, my_weth_after_higher = calculate_arb_profit((upper + mid)*10**18/2, target_token_pool, target_weth_pool, token_pool, weth_pool, pct=pct)
        lower_profit, tokens_lower, my_weth_after_lower = calculate_arb_profit((lower + mid)*10**18/2, target_token_pool, target_weth_pool, token_pool, weth_pool, pct=pct)
        if higher_profit > lower_profit:
            profit = higher_profit
            tokens = tokens_higher
            my_weth_after = my_weth_after_higher
            lower = mid
        else:
            profit = lower_profit
            tokens = tokens_lower
            my_weth_after = my_weth_after_lower
            upper = mid
    return profit, tokens, my_weth_after, mid*10**18




class Arbitrage(object):

    def __init__(self):
        self.counter = 0
        self.conn = sql.connect('main_db.db')

        # self.loop = asyncio.get_event_loop()

    def get_ticks(self):
        return asyncio.run(self.async_frontrun2())

    async def async_frontrun2(self):
        uri = "wss://api.blocknative.com/v0"
        async with websockets.connect(uri) as ws:
            res = await ws.send(json.dumps({'timeStamp': str(dt.datetime.now().isoformat()),
                                            'dappId': '48e1b25e-eb10-4e26-8a45-5eaf27f2e9a7',
                                            'version': '1',
                                            'blockchain': {
                                                'system': 'ethereum',
                                                'network': 'main'},
                                            'categoryCode': 'initialize',
                                            'eventCode': 'checkDappId'}))
            print(res)
            subscribe_msg = {
                'timeStamp': str(dt.datetime.now().isoformat()),
                'dappId': '48e1b25e-eb10-4e26-8a45-5eaf27f2e9a7',
                "categoryCode": "configs",
                'version': '1',
                'blockchain': {
                    'system': 'ethereum',
                    'network': 'main'},
                "eventCode": "put",
                "config": {
                    "scope": '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',
                    "filters": [{"_join": "OR", "terms": [{"contractCall.methodName": "swapExactETHForTokens"},
                                                          {"contractCall.methodName": "swapETHForExactTokens"}]},
                                {"value": {"gt": 500000000000000000}}
                                ],
                    "watchAddress": True
                }
            }
            res2 = await ws.send(json.dumps(subscribe_msg))
            self.nonce = w3.eth.getTransactionCount(ETH_ACCOUNT_FROM.address)
            self.pct_bribe = .98
            contract_executor = w3.eth.contract(address="<set contract address here>", abi=BUNDLE_EXECUTOR_ABI)

            while True:
                t = await ws.recv()
                data = json.loads(t)
                if 'event' not in data.keys() or 'transaction' not in data['event'].keys():
                    continue
                t = data['event']['transaction']

                if t[u'to'].lower() == uniswap_address:
                    try:
                        value = int(t[u'value'])
                        method, params = uniswap_contract.decode_function_input(t[u'input'])
                        token_address = params['path'][-1]
                        if method in ['swapExactETHForTokens', 'swapETHForExactTokens'] and token_address in c_sushi.keys():

                            l = list(map(lambda x: d[x], params['path']))
                            params['path'].remove(WETH_ERC20_TOKEN)
                            
                            target_weth_pool = c[w3.toChecksumAddress(token_address)][1].call()
                            target_token_pool = c[w3.toChecksumAddress(token_address)][0].call()

                            weth_pool = c_sushi[w3.toChecksumAddress(token_address)][1].call()
                            token_pool = c_sushi[w3.toChecksumAddress(token_address)][0].call()
                            profit, tokens, my_weth_after, my_weth  = binary_search_arbitrage(value, target_token_pool, target_weth_pool,
                                                                         token_pool, weth_pool,
                                                                         10 ** 10, 100, 0, pct=1)
                            pool1_contract = w3.eth.contract(
                                w3.toChecksumAddress(c_sushi[w3.toChecksumAddress(params['path'][-1])][0].args[0]),
                                abi=uniswap_pair_abi)
                            pool2_contract = w3.eth.contract(
                                w3.toChecksumAddress(c[w3.toChecksumAddress(params['path'][-1])][0].args[0]),
                                abi=uniswap_pair_abi)
                            if profit > 0:
                                #build_bundle(token_address, value, profit, my_weth_after, my_weth, self.nonce) for Flashbots use
                                contract_executor.functions.uniswapWeth(my_weth, 
                                                                        self.pct_bribe*profit,
                                                                        [pool1_contract.address, pool2_contract.address],
                                                                        [pool1_contract.functions.swap(my_weth, tokens, pool2_contract.address, '')._encode_transaction_data(),
                                                                         pool2_contract.functions.swap(tokens, my_weth_after, my_contract.address, '')._encode_transaction_data()],
                                                                        ).transact({'from': ETH_ACCOUNT_FROM.address})


                    except KeyError as e:
                        # traceback.print_exc()
                        print(e)
                        # print('Coin not reqistered')
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                elif t[u'to'] == sushiswap_address:
                    pass



if __name__ == '__main__':
    arb = Arbitrage()
    while True:
        a = time.time()
        try:
            l = arb.get_ticks()
        except Exception as e:
            print(f"Connection broken to feed, {str(e)}, retrying.")
