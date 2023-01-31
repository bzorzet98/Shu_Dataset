from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import PageBreak, SimpleDocTemplate, Paragraph, Spacer, Flowable, Image,Table,TableStyle

class Report():
    LEFT_MARGIN=10
    RIGTH_MARGIN=90
    BOTTOM_MARGIN=10
    UPPER_MARGIN=90
    
    def __init__(self,name_path="reporte.pdf"):
        
        self.name_path=name_path
        self.doc = SimpleDocTemplate(name_path,pagesize=A4)
        self.font_styles=getSampleStyleSheet()
        self.width,self.height = A4
        
        self.story=[]
        
        self.LEFT_MARGIN= self.width*self.LEFT_MARGIN/100
        self.RIGTH_MARGIN= self.width*self.RIGTH_MARGIN/100
        self.BOTTOM_MARGIN= self.height*self.BOTTOM_MARGIN/100
        self.UPPER_MARGIN= self.height*self.UPPER_MARGIN/100
        self.a_height=self.UPPER_MARGIN
        
        self.long_space=Spacer(1,inch)
        self.middle_space=Spacer(1,0.5*inch)
        self.space=Spacer(1,0.2*inch)
        
    def build(self):
        self.doc.build(self.story, onFirstPage=self.__first_page__, onLaterPages=self.__later_page__)
        
    def __first_page__(self,canvas, doc):
        canvas.saveState()
        canvas.setFont('Times-Roman',9)
        canvas.drawString(inch, 0.75 * inch,"Page %d" % doc.page)
        canvas.restoreState()
        
    def __later_page__(self,canvas, doc):    
        canvas.saveState()
        canvas.setFont('Times-Roman', 9)
        canvas.drawString(inch, 0.75 * inch,"Page %d" % doc.page)
        canvas.restoreState()
    
    def doc_title(self,title="report"):
        self.doc.setTitle(title)
                
    def title(self,title="AAAAA"):
        #Agregamos un titulo
        styleH=self.font_styles['Heading1']
        self.story.append(Paragraph(title,styleH))
        #self.story.append(self.space)
        
    def subtitle(self,subtitle="AAAAA"):
        #Agregamos un subtitulo
        styleH=self.font_styles['Heading2']
        self.story.append(Paragraph(subtitle,styleH))
        #self.story.append(self.space)
        
    def table(self, data):
        title_background=colors.lightslategray
        table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), title_background),
        ('FONTSIZE', (0, 0), (-1, -1), 6),
        ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)])
        rowNumb = len(data)
        for row in range(1, rowNumb):
            if row % 2 == 0:
                table_background = colors.lightgrey
            else:
                table_background = colors.whitesmoke
            table_style.add('BACKGROUND', (0, row), (-1, row), table_background)
        table=Table(data, style=table_style,spaceAfter=0.2 * inch, spaceBefore=0.2 * inch)
        self.story.append(table)
        #self.story.append(Spacer(1,inch))
    
    def set_hLine(self):
        line=MCLine(self.RIGTH_MARGIN-self.LEFT_MARGIN)
        self.story.append(line)
        #self.story.append(Spacer(1,0.2*inch))
    
    def set_paragraph(self,text="AAAAAAA"): 
        paragraph = Paragraph(text, self.font_styles['Normal'])
        self.story.append(paragraph)
        #self.story.append(self.space)
        
    def set_lSpace(self):
        self.story.append(self.long_space)
        
    def set_mSpace(self):
        self.story.append(self.middle_space)
        
    def set_sSpace(self):
        self.story.append(self.space)
        
    def set_page_break(self):
        self.story.append(PageBreak())
        
    def set_image(self,path_image,width_=0.5,heigth_=0.5):
        img=Image(path_image)
        img.drawHeight =  heigth_*self.width
        img.drawWidth = width_*self.width
        self.story.append(img)
        #self.story.append(self.middle_space)
        
    def set_logo(self,path_logo):
        img=Image(path_logo,hAlign='Right')
        img.drawHeight =  inch
        img.drawWidth = inch
        self.story.append(img)
        #self.story.append(self.middle_space)

#Clase para generar una linea horizontal recta   
class MCLine(Flowable):
    """
    Line flowable --- draws a line in a flowable
    http://two.pairlist.net/pipermail/reportlab-users/2005-February/003695.html
    """
    #----------------------------------------------------------------------
    def __init__(self, width, height=0):
        Flowable.__init__(self)
        self.width = width
        self.height = height
    #----------------------------------------------------------------------
    def __repr__(self):
        return "Line(w=%s)" % self.width
    #----------------------------------------------------------------------
    def draw(self):
        """
        draw the line
        """
        self.canv.line(0, self.height, self.width, self.height)   