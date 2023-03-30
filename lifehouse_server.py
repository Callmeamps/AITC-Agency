import os
import discord
import agency

discord_token = os.environ["DISCORD_TOKEN"]

def get_response(message: str) -> str:
    f_message = message.lower()
    
    if f_message == "$adapa":
        return agency.Adapa(message)
    
    if f_message == "$ziu":
        return agency.Ziu(message)
    
    if f_message == "$ninmah":
        return agency.Ninmah(message)
    
    if f_message == "$earl":
        return agency.Earl(message)
    
    if f_message == "$earlgpt":
        return agency.EarlGPT(message)
    
async def send_message(message, user_message, is_private):
    try:
        response = get_response(user_message)
        await message.author.send(response)if is_private else await message.channel.send(response)
    
    except Exception as e:
        print(e)

def run_discord_bot():
    TOKEN = discord_token
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    
    @client.event
    async def on_ready():
        print(f"Success! {client.user} is Live!")
        
    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        
        username = str(message.author)
        user_message = str(message.author)
        channel = str(message.author)
        
        print(f"{username}: {user_message} -{channel}")
        
        if user_message[0] == "!":
            user_message = user_message[1:]
            await send_message(message, user_message, is_private=True)
        else:
            await send_message(message, user_message, is_private=False)
    client.run(TOKEN)