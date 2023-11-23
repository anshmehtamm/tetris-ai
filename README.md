# tetris-ai
AI Agent Plays Tetris


Import 
pip install "gym[accept-rom-license, atari]"
pip install "gymnasium[accept-rom-license, atari]"


1st Technique

1. Elementary Q table storing hash of each state.
2. State is a array of RAM state of an NES device.
3. Learned Q table is compressed and stored.
4. State is 128 bytes array, each byte is an integer between 0 and 255.
5. State Space = 256*128 = 32768 states.
6. At every state there can be 5 actions which can be taken.
7. Learning over 32768 states.




https://datacrystal.romhacking.net/wiki/Tetris_(NES):RAM_map