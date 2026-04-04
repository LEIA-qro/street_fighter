console.log("Program Terminated. No more code will be executed. Setting default configurations.")

client.setwindowsize(2)        
client.invisibleemulation(false) -- Restarts visual rendering
emu.displayvsync(false)         -- Disables V-Sync to allow the emulator to run as fast as possible without being capped by the monitor's refresh rate.
emu.limitframerate(true)        -- Re-enables built-in frame rate limits to cap the emulator at a standard speed, which is important for normal gameplay and debugging.
client.displaymessages(true)  
client.SetSoundOn(true)       
client.exit()