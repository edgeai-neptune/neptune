import asyncio
import logging
import os
import sys
import time
from abc import abstractmethod

import websockets

import neptune
from neptune.common.config import BaseConfig
from neptune.common.constant import ModelType
from neptune.federated_learning.data import JobInfo, AggregationData

LOG = logging.getLogger(__name__)
MAX_SIZE_BYTE = 500 * 1024 * 1024


class AggregatorConfig(BaseConfig):
    """The config of Aggregator."""

    def __init__(self):
        BaseConfig.__init__(self)

        self.bind_ip = os.getenv("AGG_BIND_IP", "0.0.0.0")
        self.bind_port = int(os.getenv("AGG_BIND_PORT", "7363"))
        self.participants_count = int(os.getenv("PARTICIPANTS_COUNT", "1"))


class Aggregator:
    """Abstract class of aggregator"""

    def __init__(self, model_type=ModelType.GlobalModel):
        self.config = AggregatorConfig()

        self.current_round = 0
        self.agg_data_dict = {}
        self.agg_data_dict_aggregated = {}
        self.global_model = None
        self.model_type = model_type
        self.exit_flag = False
        self.current_worker = None  # specify a worker for update task info.
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    @abstractmethod
    def aggregate(self):
        """Some algorithms can be aggregated in sequence,
        but some can be calculated only after all aggregated data is uploaded.
        therefore, this abstractmethod should consider that all weights are
        uploaded.
        """
        pass

    def exit_check(self):
        """Check the current_round equal to exit_round or not."""
        exit_round = int(neptune.context.get_parameters('exit_round'))
        if self.current_round >= exit_round:
            self.exit_flag = True
            for _, agg_data in self.agg_data_dict_aggregated.items():
                agg_data.exit_flag = True

    def update_task_info(self):
        """Update the job_info information."""
        job_info = JobInfo()
        self.current_round += 1
        job_info.currentRound = self.current_round
        job_info.sampleCount = sum([x.sample_count for x in
                                    self.agg_data_dict_aggregated.values()])
        job_info.startTime = self.start_time
        job_info.updateTime = time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime())
        self.agg_data_dict_aggregated[self.current_worker].task_info = (
            job_info.to_json()
        )


class AggregationServer:
    """Websocket server that provide aggregation function."""

    def __init__(self, aggregator: Aggregator):
        super().__init__()
        self.aggregator = aggregator
        self.config = self.aggregator.config

        self.start_server = websockets.serve(self._receive,
                                             self.config.bind_ip,
                                             self.config.bind_port,
                                             max_size=MAX_SIZE_BYTE)
        self.ws_clients = {}
        LOG.info("start websocket on "
                 f"ws://{self.config.bind_ip}:{self.config.bind_port}")
        asyncio.get_event_loop().run_until_complete(self.start_server)
        asyncio.get_event_loop().run_forever()

    async def _receive(self, websocket, path):
        async for json_data in websocket:
            agg = AggregationData.from_json(json_data)
            self.aggregator.agg_data_dict[agg.worker_id] = agg
            self.ws_clients[agg.worker_id] = websocket
            if self.aggregator.current_worker is None:
                self.aggregator.current_worker = agg.worker_id
                LOG.info(f"current aggregation worker_id is {agg.worker_id}")
            LOG.info(f"received agg data, "
                     f"task_name={agg.task_id}, "
                     f"worker_id={agg.worker_id}, "
                     f"agg_data_dict length "
                     f"is {len(self.aggregator.agg_data_dict)}")

            agg_data_dict_len = len(self.aggregator.agg_data_dict)
            if agg_data_dict_len == self.config.participants_count:
                self.aggregator.aggregate()
                self.aggregator.update_task_info()
                self.aggregator.exit_check()
                items = self.aggregator.agg_data_dict_aggregated.items()
                # use coroutine the send the data
                tasks = []
                for worker_id, agg_data in items:
                    tasks.append(asyncio.create_task(
                        self.ws_clients[worker_id].send(agg_data.to_json())
                    ))
                    LOG.info("send agg_data to worker, "
                             f"worker_id = {worker_id}")

                # wait for all task complete
                for task in tasks:
                    await task

                # record the next round start time, after all agg data
                # transmitted.
                self.aggregator.start_time = time.strftime("%Y-%m-%d %H:%M:%S",
                                                           time.localtime())

                if self.aggregator.exit_flag:
                    LOG.info(f"aggregation finished, now exiting...")
                    asyncio.get_event_loop().stop()


class AggregationClient:
    """Client that interacts with the cloud aggregator."""
    _retry = 15
    _retry_interval_seconds = 3

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.agg_data = None
        self.ws_url = f"ws://{self.ip}:{self.port}"
        self.ws = None

    async def _update_weights(self, agg_data):
        error = None
        if self.ws is None:
            self.ws = await websockets.connect(self.ws_url,
                                               max_size=MAX_SIZE_BYTE)

        for i in range(AggregationClient._retry):
            try:
                await self.ws.send(agg_data.to_json())
                LOG.info(f"size of the message is {len(agg_data.to_json())}")
                result = await self.ws.recv()
                return result
            except Exception as e:
                error = e
                time.sleep(AggregationClient._retry_interval_seconds)
                self.ws = await websockets.connect(self.ws_url,
                                                   max_size=MAX_SIZE_BYTE)

        LOG.error(f"websocket error: {error}, "
                  f"retry times: {AggregationClient._retry}")
        return None

    def update_weights(self, agg_data: AggregationData):
        """Send the work's agg data to cloud aggregator, and wait
        for the updated information.

        :param agg_data: the data send to the cloud aggregator
        :return: the updated information from the cloud aggregator
        """
        LOG.info(f"start to send agg_data, current task is {agg_data.task_id}")
        loop = asyncio.get_event_loop()
        agg_data_json = loop.run_until_complete(self._update_weights(agg_data))
        if agg_data_json is None:
            LOG.error("send data to agg worker failed, exist worker")
            sys.exit(1)
        agg_data = AggregationData.from_json(agg_data_json)
        LOG.info(f"received result, worker_id = {agg_data.worker_id}")
        return agg_data
