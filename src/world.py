from __future__ import print_function

from builtins import range
from malmo import MalmoPython
import os
import sys
import time
import json
import numpy as np



class World:
    """
    Create the Malmo environment for a fight arena filled with zombies
    """

    def __init__(self, size, obs_size,num_entities, episodes = 100):
        self.size = size
        self.obs_size = obs_size
        self.num_entities = num_entities
        self.episodes = episodes

        # Create default Malmo objects:

        #the agent
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print(self.agent_host.getUsage())
            exit(0)


        self.world_state = self.agent_host.getWorldState()
        self.is_mission_running = self.world_state.is_mission_running


    def drawWall(self,blocktype, height) -> str:
        """

        :param blocktype: the type of block wanted to put
        :param height: the height of the wall
        :return:
        """
        genString = ""

        for i in range(1, height + 1):
            genString += '<DrawLine type="' + blocktype + '" y1="' + str(i) + '" y2="' + str(
                i) + '" x1="-'+str(self.size)+ '" x2="'+str(self.size)+ '" z1="-'+str(self.size)+ '" z2="-'+str(self.size)+ '" />'
            genString += '<DrawLine type="' + blocktype + '" y1="' + str(i) + '" y2="' + str(
                i) + '" x1="-'+str(self.size)+ '" x2="-'+str(self.size)+ '" z1="-'+str(self.size)+ '" z2="'+str(self.size)+ '" />'
            genString += '<DrawLine type="' + blocktype + '" y1="' + str(i) + '" y2="' + str(
                i) + '" x1="'+str(self.size)+ '" x2="'+str(self.size)+ '" z1="-'+str(self.size)+ '" z2="'+str(self.size)+ '" />'
            genString += '<DrawLine type="' + blocktype + '" y1="' + str(i) + '" y2="' + str(
                i) + '" x1="-'+str(self.size)+ '" x2="'+str(self.size)+ '" z1="'+str(self.size)+ '" z2="'+str(self.size)+ '" />'

        return genString


    def drawEntity(self,entity) -> str:
        """

        :param num: number of zombies wanted present in the world
        :return: a string representing the zombies
        """

        genString = ""

        for i in range(self.num_entities):
            x = np.random.randint(-self.size,self.size)
            z = np.random.randint(-self.size, self.size)

            genString+='<DrawEntity x="'+ str(x) +'" y="7" z="'+ str(z) +'" type="'+entity + '"/>'

        return genString

    def GetMissionXML(self):

        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                  <About>
                    <Summary>Kill Zombies</Summary>
                  </About>

                  <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>9000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    <AllowSpawning> false </AllowSpawning>
                    </ServerInitialConditions>
                    <ServerHandlers>
                      <FlatWorldGenerator generatorString="3;7,2;1;"/>
                        <DrawingDecorator>
                            ''' + self.drawWall("cobblestone_wall", 10) + \
                                self.drawEntity("Zombie") + '''
                        </DrawingDecorator>

                      <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                  </ServerSection>

                  <AgentSection mode="Survival">
                    <Name>Serious Sam</Name>
                    <AgentStart>
                        <Placement x="0" y="7" z="0"/>
                        <Inventory>
                            <InventoryItem slot="0" type="golden_sword"/>
                        </Inventory>
                    </AgentStart>

                    <AgentHandlers>
                        <RewardForDamagingEntity>
                            <Mob type="Zombie" reward="100"/>
                        </RewardForDamagingEntity>
                        <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-''' + str(int(self.obs_size / 2)) + '''" y="-1" z="-''' + str(
            int(self.obs_size / 2)) + '''"/>
                                    <max x="''' + str(int(self.obs_size / 2)) + '''" y="7" z="''' + str(int(self.obs_size / 2)) + '''"/>
                                </Grid>
                            </ObservationFromGrid>
                        <ObservationFromRay/>
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="100" yrange="2" zrange="100" update_frequency="1"/>
                        </ObservationFromNearbyEntities>
                      <ObservationFromFullStats/>
                      <DiscreteMovementCommands/>
                      <AgentQuitFromReachingCommandQuota total="'''+str(self.episodes)+'''" />
                    </AgentHandlers>
                  </AgentSection>
                </Mission>'''


    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.GetMissionXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "DiamondCollector")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        return self.agent_host


    def get_world_state(self):
        """
        get the state of the world
        :return: world state
        """

        self.world_state = self.agent_host.getWorldState()
        self.is_mission_running = self.world_state.is_mission_running

        return self.world_state

    def get_reward(self, death, episode_steps):
        """

        :return: integer of the reward the agent received
        """

        reward = 0
        for r in self.world_state.rewards:
            add = r.getValue()
            if add == 100:
                print("hit zombie")
            reward += add
        if death:
            reward += -1000
        if episode_steps >= self.episodes:
            reward += -100
        return reward



    def get_observation(self):
        """
        Use the agent observation API to get information on the zombie's locations.
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: a 2d array
        """
        #only look a obs_size by obs_sive around the agent
        obs = np.zeros((self.obs_size, self.obs_size), dtype = np.float32)
        count = 0
        life = 0

        while self.is_mission_running:
            time.sleep(0.1)
            self.world_state = self.get_world_state()
            if len(self.world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if self.world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = self.world_state.observations[-1].text
                observations = json.loads(msg)
                #print("observations", observations)
                #print("observations full stats", observations[u'Life'])

                # current location of the agent
                # which will be center of the observation matrix
                xpos, ypos, zpos = observations[u'XPos'], observations[u'YPos'], observations[u'ZPos']
                yaw = observations[u'Yaw']
                life = observations[u'Life']
                #print("current", xpos,zpos, yaw)

                halfway = self.obs_size//2

                # Get observation with location of all the zombies
                entities = observations[u'entities']
                #print("entitites", entities)
                for e in entities:

                    if e['name'] == 'Zombie':
                        count += 1
                        x = int(e['x'])
                        z = int(e['z'])


                        if abs(x-xpos) <= halfway and abs(z-zpos) <= halfway:
                             i = x - xpos + halfway
                             j = z - zpos + halfway
                             #print("zombie location", x, z, i , j)

                             #had to flip i and j to match row and column to the x,z coords in malmo
                             obs[int(j)][int(i)] = 1


                # need to fix the rotation of the observation
                # axes of the observation are 1-d
                # rotate it so the zombie in front of him is in front of him if we look at the matrix

                # Rotate observation with orientation of agent

                if yaw == 270:
                    obs = np.rot90(obs, k=1)
                elif yaw == 180:
                    obs = np.rot90(obs, k=2)
                elif yaw == 90:
                    obs = np.rot90(obs, k=3)
                #print("observation", obs)
                break


        return obs,count,life
