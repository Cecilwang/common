\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{slide}[2021/05/25 CecilWang's Slide]

\LoadClass{beamer}
\RequirePackage{common}

%\usetheme{default}
%\usetheme{AnnArbor}
%\usetheme{Antibes}
%\usetheme{Bergen}
%\usetheme{Berkeley}
%\usetheme{Berlin}
%\usetheme{Boadilla}
\usetheme{CambridgeUS}
%\usetheme{Copenhagen}
%\usetheme{Darmstadt}
%\usetheme{Dresden}
%\usetheme{Frankfurt}
%\usetheme{Goettingen}
%\usetheme{Hannover}
%\usetheme{Ilmenau}
%\usetheme{JuanLesPins}
%\usetheme{Luebeck}
%\usetheme{Madrid}
%\usetheme{Malmoe}
%\usetheme{Marburg}
%\usetheme{Montpellier}
%\usetheme{PaloAlto}
%\usetheme{Pittsburgh}
%\usetheme{Rochester}
%\usetheme{Singapore}
%\usetheme{Szeged}
%\usetheme{Warsaw}

%\usecolortheme{albatross}
%\usecolortheme{beaver}
%\usecolortheme{beetle}
%\usecolortheme{crane}
%\usecolortheme{dolphin}
%\usecolortheme{dove}
%\usecolortheme{fly}
%\usecolortheme{lily}
%\usecolortheme{orchid}
%\usecolortheme{rose}
%\usecolortheme{seagull}
%\usecolortheme{seahorse}
%\usecolortheme{whale}
%\usecolortheme{wolverine}

% Color Override
\setbeamercolor{background canvas}{bg=LaughGray}

\setbeamercolor{author in head/foot}{fg=White,bg=PKURed}
\setbeamercolor{title in head/foot}{fg=White,bg=AuxiliaryColor1}
\setbeamercolor{date in head/foot}{fg=White,bg=AuxiliaryColor2}

\setbeamercolor{section in head/foot}{fg=White,bg=PKURed}
\setbeamercolor{subsection in head/foot}{fg=White,bg=AuxiliaryColor2}

\setbeamercolor{title}{fg=White,bg=PKURed}
\setbeamercolor{frametitle}{fg=PKURed,bg=LaughGray}

\setbeamercolor{section number projected}{fg=White,bg=PKURed}
\setbeamercolor{item projected}{fg=White,bg=PKURed}

\setbeamercolor{block title}{fg=PKURed,bg=DeepLaughGray}
\setbeamercolor{block body}{fg=Black,bg=DeepLaughGray}

\captionsetup[table]{labelfont={color=Black}}

% Footline
%\setbeamertemplate{footline}
%\setbeamertemplate{footline}[page number]
%\setbeamertemplate{navigation symbols}{}

% Overview
\AtBeginSection[]{
  \begin{frame}
    \frametitle{Overview}
    \tableofcontents[currentsection]
  \end{frame}
}

\setcounter{tocdepth}{1}

% Command
\newcommand{\Begin}[6]{
  \title[#3]{#2}

  \author{#1}

  \institute[#5] {
    #4 \\\medskip
    \textit{#6}
  }

  \date{\today}

  \begin{document}

  \begin{frame}
    \titlepage
  \end{frame}

  \begin{frame}
    \frametitle{Overview}
    \tableofcontents
  \end{frame}
}
\newcommand{\End}{
  \begin{frame}
    \Huge{\centerline{The End}}
  \end{frame}

  \end{document}
}
