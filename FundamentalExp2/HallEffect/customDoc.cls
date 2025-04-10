%% This is file `custom.cls'.

%%------------------ 文档类声明, 与文件名一致 ------------------
\ProvidesClass{customDoc}[custom class for physnya]
%%------------------------------------------------------------

\LoadClass[12pt]{article}

\usepackage{ctex}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{amsfonts, amssymb, amsthm, amsmath, extarrows}
\usepackage{physics}
\usepackage{multirow, makecell}
\usepackage{titlesec} % 用于自定义标题格式

\RequirePackage{fontspec}
% 中文字体设置：CJK —— 共6个 √
% BoldFont：粗体  ItalicFont：斜体 √
% {} 内是字体包名字，[] 是细节参数的设置 √
\setCJKmainfont{LXGWWenKai-Light.ttf}[BoldFont={LXGWWenKai-Light.ttf}, 
                                    ItalicFont={LXGWWenKai-Light.ttf}, 
                                    BoldItalicFont={LXGWWenKai-Light.ttf}]  % 默认字体 √
\setCJKsansfont{LXGWWenKai-Light.ttf}   % 无衬线字体：同英文字体设置 √
\setCJKmonofont{LXGWWenKai-Light.ttf}  % 等宽字体：同英文字体设置 √

% 公式编号格式设置
\numberwithin{equation}{section}

% 超链接格式
\usepackage[hidelinks, colorlinks = true, linkcolor = teal, citecolor = green]{hyperref}

% 页面基本布局, 页边距, 页眉样式
\usepackage{geometry, fancyhdr}
\usepackage{lastpage}
\geometry{
    a4paper,
    left = 27mm,
    right = 27mm,
    top = 25.4mm, 
    bottom = 25.4mm,
}
\fancypagestyle{fancy}{
    \fancyhf{}
    \lhead{\the\year 年 \the\month 月}
    \chead{\centering\@title}
    \rhead{第 \thepage\ 页\ \ 共 \pageref{LastPage}\ 页}
    \lfoot{}
    \cfoot{}
    \rfoot{}
}
\pagestyle{fancy}

% 定义变量, 储存班级姓名学号
\def\title#1{\gdef\@title{#1}}
\def\class#1{\gdef\@class{#1}}
\def\name#1{\gdef\@name{#1}}
\def\id#1{\gdef\@id{#1}}
\class{[班级]}\name{[姓名]}\id{[学号]}

% 标题页修改
\renewcommand\maketitle{
    \par
    \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1em{%
            \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
            \newpage
            \global\@topnum\z@   % Prevents figures from going at top of page.
        \@maketitle
        % \thispagestyle{plain}
        \@thanks
    \endgroup
    \setcounter{footnote}{0}
    \global\let\thanks\relax
    \global\let\maketitle\relax
    \global\let\@maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    % \global\let\@title\@empty
    \global\let\title\relax
    \global\let\class\relax
    \global\let\date\relax
    \global\let\and\relax
}
% 标题格式
\def\@maketitle{
    \newpage
    \null
    \vskip 0.8em
    \begin{center}
    \let \footnote \thanks
    {\LARGE \@title \par}
    \vskip 1.2em
    {\kaishu
        \@class\ \ \ \@name\ \ \ \@id
        \par}
    \end{center}
    \par
    \vskip 0.8em
}

% 设置 section 标题格式，标号为普通字体
\titleformat{\section}
  {\normalfont\Large} % 标题样式：普通字体，字号为 Large
  {\thesection}       % 标号样式
  {1em}               % 标号与标题之间的间距
  {}

% 设置 subsection 标题格式，标号为普通字体
\titleformat{\subsection}
  {\normalfont\large} % 标题样式：普通字体，字号为 large
  {\thesubsection}    % 标号样式
  {1em}               % 标号与标题之间的间距
  {}

% 插入图片(构建了语法糖)    用法: \picture[scale]{file}[caption][label]
%% 基本情况
\newcommand{\picture@helper@basic}[2]{
    \begin{figure}[ht]\centering
        \includegraphics[width=#1\columnwidth]{#2}
    \end{figure}}
%% 有标题的情况
\newcommand{\picture@helper@caption}[3]{
    \begin{figure}[ht]\centering
        \includegraphics[
            width=#1\columnwidth
            ]
        {#2}\caption{#3}
    \end{figure}
}
%% 有标签的情况
\newcommand{\picture@helper@label}[4]{
    \begin{figure}[ht]\centering
        \includegraphics[
            width=#1\columnwidth
            ]
        {#2}\caption{#3}\label{#4}
    \end{figure}
}
%% 选择器
\def\picture@pick@caption#1#2[#3]{
    \@ifnextchar[%
    {\picture@pick@label{#1}{#2}{#3}}%
    {\picture@helper@caption{#1}{#2}{#3}}%
}
\def\picture@pick@label#1#2#3[#4]{\picture@helper@label{#1}{#2}{#3}{#4}}
\def\picture@helper@width [#1]#2{%
    \@ifnextchar[%
    {\picture@pick@caption{#1}{#2}}%
    {\begin{figure}[ht]
        \centering
        \includegraphics[
            width=#1\columnwidth
            ]{#2}
    \end{figure}}%
}

%% 基本命令
\renewcommand{\picture}{
    \@ifnextchar[{\picture@helper@width}
    {\picture@helper@width[0.7]}
}

\setlength{\lineskiplimit}{6pt}
\setlength{\lineskip}{6pt}
\setlength{\abovedisplayskip}{6pt}   % 公式上下间隔设置为6pt比较好看，可以自己更改
\setlength{\belowdisplayskip}{6pt}
\setlength{\abovedisplayshortskip}{6pt}
\setlength{\belowdisplayshortskip}{6pt} 

% (未)实现表格分栏, 参考https://tex.stackexchange.com/questions/555318/environment-to-balance-a-table-in-a-multicol