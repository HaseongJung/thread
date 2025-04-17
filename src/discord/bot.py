import os
from dotenv import load_dotenv, find_dotenv
import discord
from discord.ext import commands
from command import Command

print(os.getcwd())

# Load Discord Bot token
load_dotenv('./')
bot_token = os.environ.get('DISCORD_BOT_TOKEN')


# Initialize Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} - {bot.user.id}')
    print('------')
    await bot.add_cog(Command(bot))

if __name__ == "__main__":
    # Run the bot
    bot.run(bot_token)

