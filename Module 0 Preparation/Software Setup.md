# Software Setup

今年，建议的工作方式是通过[Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)。但是，如果您已经拥有支持GPU的硬件，并且希望在本地工作，我们将为您提供设置虚拟环境的说明。

- [在Google合作实验室上远程工作](#在Google合作实验室上远程工作)
- 在您的机器上本地工作
  - [Anaconda虚拟环境](#Anaconda虚拟环境)
  - [Python venv](#python-venv)
  - [安装软件包](#安装软件包)

## 在Google合作实验室上远程工作

Google合作实验室基本上是Jupyter笔记本和Google云端硬盘的组合。它完全在云中运行，并预装了许多软件包（例如PyTorch和Tensorflow），因此每个人都可以访问相同的依赖项。更酷的一点是，Colab受益于免费访问硬件加速器（例如GPU（K80，P100）和TPU）的优势，这对于任务2和3尤其有用。

**要求**。要使用Colab，您必须拥有一个具有关联的Google云端硬盘的Google帐户。假设您同时拥有两者，则可以通过以下步骤将Colab连接到云端硬盘：

1. 单击右上角的滚轮，然后选择`Settings`.
2. 单击`Manage Apps` tab
3. 在顶部，选择 `Connect more apps` ，这将打开一个GSuite Marketplace窗口。
4. 搜索Colab，然后单击添加

**工作流程**。每个作业都为您提供一个下载链接，该链接指向包含Colab笔记本和Python入门代码的zip文件。您可以将文件夹上传到云端硬盘，在Colab中打开笔记本并对其进行处理，然后将进度保存回云端硬盘。我们建议您观看下面的教程视频，其中以推荐作业1为例介绍推荐的工作流程。



<iframe width="560" height="315" src="https://www.youtube.com/embed/IZUz4pRYlus" frameborder="0" allowfullscreen="" style="margin: auto; padding: 0px; color: rgb(0, 0, 0); font-family: Roboto, sans-serif; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 300; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: block;"></iframe>

**最佳做法**。使用Colab时，您需要注意一些事项。首先要注意的是，不能保证资源（这是免费的代价）。如果您闲置了一段时间或您的总连接时间超过了允许的最大时间（〜12小时），则Colab VM将断开连接。这意味着所有未保存的进度都将丢失。因此，养成在分配工作时经常保存代码的习惯。要了解有关Colab中资源限制的更多信息，请在此处阅读其常见问题[解答](https://research.google.com/colaboratory/faq.html)。

**使用GPU**。使用GPU就像在Colab中切换运行时一样简单。具体来说，单击`Runtime -> Change runtime type -> Hardware Accelerator -> GPU`，GPU计算将自动支持您的Colab实例。

如果您有兴趣了解有关Colab的更多信息，建议您访问以下资源：

- [Google Colab简介](https://www.youtube.com/watch?v=inN8seMm7UI)
- [Welcome to Colab](https://colab.research.google.com/notebooks/intro.ipynb)
- [Overview of Colab Features](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)



## 在您的机器上本地工作

如果您希望在本地工作，则应使用虚拟环境。您可以通过Anaconda（推荐）或通过Python的本地`venv`模块安装一个。确保您使用的是Python 3.7，**因为我们不再支持Python 2**。

### Anaconda虚拟环境

我们强烈建议您使用免费的Anaconda Python发行版，该发行版为您处理软件包依赖项提供了一种简便的方法。请确保下载当前安装Python 3.7的Python 3版本。 Anaconda的整洁之处在于，默认情况下它附带了MKL优化，这意味着您的`numpy`和`scipy`代码可从显着的加速中受益，而无需更改任何代码行。

由于已安装Anaconda，因此为课程创建虚拟环境很有意义。如果您选择不使用虚拟环境（强烈建议不要使用！），则由您决定是否将代码的所有依赖项全局安装在您的计算机上。要设置名为`cs231n`的虚拟环境，请在终端中运行以下命令：

```
# this will create an anaconda environment
# called cs231n in 'path/to/anaconda3/envs/'
conda create -n cs231n python=3.7
```

要激活并进入环境，请运行`conda activate cs231n`。要禁用环境，请运行`conda deactivate cs231n`或退出终端。请注意，每次您要进行分配时，都应重新运行`conda activate cs231n`。

```
# sanity check that the path to the python
# binary matches that of the anaconda env
# after you activate it
which python
# for example, on my machine, this prints
# $ '/Users/kevin/anaconda3/envs/sci/bin/python'
```

您可以参考此[页面](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)以获取有关使用Anaconda管理虚拟环境的更多详细说明。

**注意：**如果您选择了“ Anaconda”路线，则可以安全地跳过下一部分，直接进入[安装软件包]()。

### Python venv

从3.3版本开始，Python随附了一个名为venv的轻量级虚拟环境模块。每个虚拟环境都打包了自己独立的一组已安装的Python软件包，这些软件包与系统范围的Python软件包隔离，并且运行的Python版本与用于创建它的二进制版本匹配。要设置名为`cs231n`的虚拟环境，请在终端中运行以下命令：

```
# this will create a virtual environment
# called cs231n in your home directory
python3.7 -m venv ~/cs231n
```

要激活并进入环境，请运行`source〜/ cs231n / bin / activate`。要禁用环境，请运行 `deactivate` 或退出终端。请注意，每次您要进行分配时，都应重新运行 `source ~/cs231n/bin/activate`。

```
 sanity check that the path to the python
# binary matches that of the virtual env
# after you activate it
which python
# for example, on my machine, this prints
# $ '/Users/kevin/cs231n/bin/python'
```

### 安装软件包

设置并激活虚拟环境（通过`conda`或`venv`）后，您应该使用`pip`安装运行任务所需的库。为此，请运行：

```
# again, ensure your virtual env (either conda or venv)
# has been activated before running the commands below
cd assignment1  # cd to the assignment directory

# install assignment dependencies.
# since the virtual env is activated,
# this pip is associated with the
# python binary of the environment
pip install -r requirements.txt
```

