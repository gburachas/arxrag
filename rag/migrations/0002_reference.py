from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):
    dependencies = [
        ('rag', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Reference',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('raw_text', models.TextField()),
                ('arxiv_id', models.CharField(blank=True, db_index=True, max_length=32)),
                ('title', models.TextField(blank=True)),
                ('authors', models.TextField(blank=True)),
                ('position', models.IntegerField(default=0)),
                ('vector', models.BinaryField(blank=True, null=True)),
                ('added_at', models.DateTimeField(auto_now_add=True)),
                ('doc', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='rag.document')),
            ],
        ),
    ]
