import os
import discord
from discord.ext import commands
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()





def get_latest_result_dir():
    """
    Get the latest result directory based on the naming convention.
    """
    output_dir = "./output/"
    # Sort directories by date and time
    latest_dir = sorted(os.listdir(output_dir))[-1]
    return os.path.join(output_dir, latest_dir)





class Command(commands.Cog):
    def __init__(self, bot):
        self.bot = bot


    @commands.command(name='ping')
    async def ping(self, ctx: discord.ext.commands.Context):
        '''
        Check if the bot is alive
        '''
        await ctx.send('Pong!')


    @commands.command(name='run')
    async def run(self, ctx: discord.ext.commands.Context):
        '''
        Run the bash script to collect news data and perform topic modeling.
        '''
        # Run the bash script
        await ctx.send('Get News data & Topic modeling...')
        # subprocess.call("bash work.sh", shell=True)
        returncode = await asyncio.to_thread(subprocess.call, "bash work.sh", shell=True)
        if returncode == 0:
            await ctx.send("✅ 작업이 정상 종료되었습니다.")
        else:
            await ctx.send(f"❌ 작업이 비정상 종료되었습니다 (exit {returncode})")

        # loop = asyncio.get_event_loop()
        # last_result_dir = await loop.run_in_executor(executor, get_latest_result_dir)
        last_result_dir = get_latest_result_dir()
        await ctx.send(f"가장 최신 결과: {last_result_dir}")
        # await ctx.send('Command executed successfully!')  # Replace with actual result if needed


    @commands.command(name='get_result')
    async def file_test(slef, ctx: discord.ext.commands.Context):
        '''
        Get the latest result of topic modeling
        '''
        # Load the latest result directory
        output_dir = "./output/"
        last_result_dir = sorted(os.listdir(output_dir))[-1]
        await ctx.send(f"날짜: {last_result_dir.split('_')[0]}, 시간: {last_result_dir.split('_')[1]}")


        # Send the Chart images
        distribution_chart = os.path.join(output_dir, last_result_dir, "Chart", "Document_Distribution.png")
        probability_chart = os.path.join(output_dir, last_result_dir, "Chart", "Topic_Probability_Distribution.png")
        await ctx.send(file=discord.File(distribution_chart))
        await ctx.send(file=discord.File(probability_chart))

        # Send the CSV files
        documents = sorted(os.listdir(os.path.join(output_dir, last_result_dir, "Documents")))
        for i, name in enumerate(documents):
            document_path = os.path.join(output_dir, last_result_dir, "Documents", name)
            await ctx.send(name)    # send file name
            await ctx.send(file=discord.File(document_path))    # send file


    @commands.command(name='get_posts')
    async def get_posts(self, ctx: discord.ext.commands.Context):
        '''
        Get the latest result of generated posts
        '''
        # Load the latest result directory
        output_dir = "./output/"
        last_result_dir = sorted(os.listdir(output_dir))[-1]
        
        # Send the latest result directory
        await ctx.send(f"날짜: {last_result_dir.split('_')[0]}, 시간: {last_result_dir.split('_')[1]}")

        # Send the generated posts
        generated_posts = sorted(os.listdir(os.path.join(output_dir, last_result_dir, "Posts")))
        await ctx.send(f"생성된 포스트 수: {len(generated_posts)}")
        for i, name in enumerate(generated_posts):
            post_path = os.path.join(output_dir, last_result_dir, "Posts", name)
            await ctx.send(name)
            await ctx.send(file=discord.File(post_path))

        


